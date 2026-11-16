# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare WNA16 tiles for Qwen4Exp FP16 decode on one idle MI100.

Runs the real GPTQ/AWQ kernels, activation, and expert reduction with synthetic
weights at the model's per-rank dimensions. Timing is hot-cache CUDA graph time.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

from vllm.model_executor.layers.fused_moe import override_config
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.accelerator.set_device_index(0)
    if any(m < 1 or m > 5 for m in args.rows):
        parser.error("decode rows must be between 1 and 5")
    E, H, intermediate = 512, 2560, 160
    w1 = torch.randint(
        0, 256, (E, intermediate * 2, H // 2), dtype=torch.uint8, device="cuda"
    )
    w2 = torch.randint(
        0, 256, (E, H, intermediate // 2), dtype=torch.uint8, device="cuda"
    )
    s1 = torch.full(
        (E, intermediate * 2, H // 32), 0.02, dtype=torch.float16, device="cuda"
    )
    s2 = torch.full(
        (E, H, intermediate // 32), 0.02, dtype=torch.float16, device="cuda"
    )
    configs = [("default", None)]
    for n, warps, waves, kpack in [
        (16, 1, 0, 1),
        (16, 1, 2, 2),
        (32, 2, 2, 1),
        (32, 4, 2, 2),
        (64, 4, 2, 1),
        (64, 4, 4, 2),
        (32, 4, 0, 1),
        (64, 8, 2, 1),
    ]:
        configs.append(
            (
                f"n{n}-w{warps}-e{waves}-k{kpack}",
                dict(
                    BLOCK_SIZE_M=16,
                    BLOCK_SIZE_N=n,
                    BLOCK_SIZE_K=32,
                    GROUP_SIZE_M=1,
                    SPLIT_K=1,
                    num_warps=warps,
                    num_stages=2,
                    waves_per_eu=waves,
                    matrix_instr_nonkdim=16,
                    kpack=kpack,
                ),
            )
        )
    rows = []
    for m in args.rows:
        x = torch.randn(m, H, device="cuda", dtype=torch.float16) * 0.1
        ids = torch.stack([torch.randperm(E, device="cuda")[:10] for _ in range(m)]).to(
            torch.int32
        )
        weights = torch.softmax(torch.randn(m, 10, device="cuda"), -1)

        def run(x=x, weights=weights, ids=ids):
            return fused_experts_impl(
                x,
                w1,
                w2,
                weights,
                ids,
                use_int4_w4a16=True,
                w1_scale=s1,
                w2_scale=s2,
                block_shape=[0, 32],
            )

        ref = run()
        for label, cfg in configs:
            with override_config(cfg):
                result = run()
                torch.accelerator.synchronize()
                if label == "n16-w1-e0-k1":
                    torch.testing.assert_close(result, ref, rtol=0, atol=0)
                else:
                    torch.testing.assert_close(result, ref, rtol=0.005, atol=0.002)
                for _ in range(3):
                    run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(8):
                        run()
                samples = []
                for _ in range(5):
                    a, b = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    a.record()
                    for _ in range(100):
                        graph.replay()
                    b.record()
                    b.synchronize()
                    samples.append(a.elapsed_time(b) * 1000 / 800)
                row = dict(
                    m=m,
                    label=label,
                    config=cfg,
                    us=statistics.median(samples),
                    max_abs_error=(result - ref).abs().max().item(),
                )
                rows.append(row)
                print(row, flush=True)
            args.output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
