# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check and time MI100 push collectives with four torchrun processes."""

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


def compare(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo")
    assert dist.get_world_size() == 4
    ca = CustomAllreduce(dist.group.WORLD, rank)
    assert not ca.disabled, "native custom AR unavailable"
    push = ca._push_ar
    assert push is not None, "push AR unavailable; set VLLM_GFX908_PUSH_AR=1"
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    torch.manual_seed(8 + rank)
    checks = 0
    try:
        for shape in [(1, 2560), (4, 2560), (16, 2560), (48, 2560), (2, 2, 2560)]:
            x = torch.randn(shape, dtype=dtype, device="cuda")
            before = push.calls
            actual = ca.custom_all_reduce(x)
            assert actual is not None
            assert push.calls > before, f"{dtype} input fell back to native AR"
            expected = ca.all_reduce(x, registered=False)
            compare(actual, expected)
            checks += 1

        # Include the sentinel value, subnormals, overflow, infinity, and NaN.
        values = torch.tensor(
            [
                0.0,
                -0.0,
                2**-24,
                -(2**-24),
                65504.0,
                -65504.0,
                float("inf"),
                float("nan"),
            ],
            dtype=dtype,
            device="cuda",
        )
        edge = values.repeat(320).view(1, 2560)
        compare(ca.custom_all_reduce(edge), ca.all_reduce(edge, registered=False))
        checks += 1

        # Exercise both typed residual consumers through the real HIP entry point.
        from vllm.distributed.device_communicators.gfx908_push_ar import _ext

        residual = torch.randn(4, 2560, dtype=dtype, device="cuda")
        partial = torch.randn_like(residual)
        residual_out = torch.empty_like(residual)
        ext = _ext()
        ext.push(partial, push.ptrs, rank * push.slot_elems)
        ext.consume_res(
            residual_out,
            residual,
            push.ptrs[rank],
            push.slot_elems,
            push.stats,
            push.max_spin,
            0,
            0,
        )
        compare(residual_out, ca.all_reduce(partial, registered=False) + residual)
        checks += 1

        x = torch.zeros(4, 2560, dtype=dtype, device="cuda")
        graphs = []
        for sites in [8, 12]:
            g = torch.cuda.CUDAGraph()
            with ca.capture(), torch.cuda.graph(g):
                outputs = [ca.custom_all_reduce(x) for _ in range(sites)]
            graphs.append((g, outputs))
        for i in range(args.replays):
            # Exactly representable rank-distinct values expose stale peer data.
            x.fill_((rank + 1) * ((i % 31) - 15) / 128)
            if i % 7 == 0:
                compare(ca.custom_all_reduce(x), ca.all_reduce(x, registered=False))
            g, outputs = graphs[i % len(graphs)]
            g.replay()
            expected = ca.all_reduce(x, registered=False)
            for out in outputs:
                compare(out, expected)
            checks += len(outputs)

        # Queue changing inputs and alternating graphs without a host rendezvous
        # between replays. This stresses slot reuse when ranks get ahead.
        for window in range(8):
            for j in range(256):
                i = window * 256 + j
                step = ((i % 31) - 15) / 128
                x.fill_((rank + 1) * step)
                g, outputs = graphs[i % len(graphs)]
                g.replay()
            expected = torch.full_like(x, 10 * step)
            for out in outputs:
                compare(out, expected)
            checks += len(outputs)
        stats = push.stats_dict()
        assert stats["timeouts"] == 0, stats
        result = {
            "dtype": args.dtype,
            "checks": checks,
            "replays": args.replays,
            "stats": stats,
            "benchmarks": [],
        }

        if args.benchmark:
            for rows in [1, 4, 16, 48]:
                inp = torch.randn(rows, 2560, dtype=dtype, device="cuda")
                runs = []
                for name in ["native", "push"]:
                    graph = torch.cuda.CUDAGraph()
                    with ca.capture(), torch.cuda.graph(graph):
                        for _ in range(8):
                            if name == "native":
                                ca.all_reduce(inp, registered=False)
                            else:
                                ca.custom_all_reduce(inp)
                    runs.append((name, graph))
                samples = {"native": [], "push": []}
                for trial in range(5):
                    for name, graph in runs if trial % 2 == 0 else list(reversed(runs)):
                        for _ in range(10):
                            graph.replay()
                        torch.accelerator.synchronize()
                        dist.barrier()
                        start, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        start.record()
                        for _ in range(200):
                            graph.replay()
                        end.record()
                        end.synchronize()
                        us = torch.tensor(
                            start.elapsed_time(end) * 1000 / 1600, dtype=torch.float64
                        )
                        dist.all_reduce(us, op=dist.ReduceOp.MAX)
                        samples[name].append(us.item())
                row = {
                    "rows": rows,
                    **{k + "_us": statistics.median(v) for k, v in samples.items()},
                    "samples": samples,
                }
                result["benchmarks"].append(row)
        assert push.stats_dict()["timeouts"] == 0
        if rank == 0:
            print(json.dumps(result, indent=2), flush=True)
            if args.output:
                args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        torch.accelerator.synchronize()
        dist.barrier()
        ca.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
