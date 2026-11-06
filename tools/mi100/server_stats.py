#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Display live token and speculative-decoding statistics from /metrics."""

import argparse
import math
import sys
import time
from datetime import datetime
from urllib.error import URLError
from urllib.request import urlopen

import regex as re

METRICS = {
    "prompt_tokens_total",
    "generation_tokens_total",
    "spec_decode_num_drafts_total",
    "spec_decode_num_draft_tokens_total",
    "spec_decode_num_accepted_tokens_total",
    "num_requests_running",
    "num_requests_waiting",
    "kv_cache_usage_perc",
    "gpu_cache_usage_perc",
    "request_success_total",
}
LINE = re.compile(r"^vllm:(\w+)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?$")
ERROR_LABEL = re.compile(r'(?:^|,)\s*finished_reason="error"(?:,|$)')


def fetch(url: str) -> dict[str, float]:
    values: dict[str, float] = {}
    with urlopen(url, timeout=10) as response:
        body = response.read().decode()
    for line in body.splitlines():
        match = LINE.fullmatch(line)
        if match is None or match[1] not in METRICS:
            continue
        name, labels, raw = match.groups()
        if name == "request_success_total" and not ERROR_LABEL.search(labels or ""):
            continue
        value = float(raw)
        if not math.isfinite(value):
            continue
        if name.endswith("cache_usage_perc"):
            values[name] = max(values.get(name, 0), value)
        else:
            values[name] = values.get(name, 0) + value
    if "generation_tokens_total" not in values:
        raise ValueError("Endpoint did not return vLLM token metrics")
    return values


def number(value: float | None, decimals: int = 0) -> str:
    return "n/a" if value is None else f"{value:,.{decimals}f}"


def ratio(values: dict[str, float], numerator: str, denominator: str) -> float | None:
    top, bottom = values.get(numerator), values.get(denominator)
    return top / bottom if top is not None and bottom else None


def display(
    url: str,
    current: dict[str, float],
    previous: dict[str, float] | None,
    elapsed: float,
) -> None:
    reset = previous is not None and any(
        current.get(key, 0) < previous.get(key, 0)
        for key in METRICS
        if key.endswith("_total")
    )
    if reset:
        previous = None
    rates = {}
    for key in ("prompt_tokens_total", "generation_tokens_total"):
        rates[key] = (
            (current[key] - previous[key]) / elapsed
            if previous is not None and key in current and key in previous
            else None
        )
    acceptance = ratio(
        current,
        "spec_decode_num_accepted_tokens_total",
        "spec_decode_num_draft_tokens_total",
    )
    accepted_per_step = ratio(
        current, "spec_decode_num_accepted_tokens_total", "spec_decode_num_drafts_total"
    )
    cache = current.get("kv_cache_usage_perc", current.get("gpu_cache_usage_perc"))
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")
    print(f"vLLM statistics — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{url}\n")
    print(
        f"Generation       {number(rates['generation_tokens_total'], 1):>14} tokens/s"
    )
    print(f"Prompt processing {number(rates['prompt_tokens_total'], 1):>13} tokens/s")
    print(
        f"Total generated  {number(current.get('generation_tokens_total')):>14} tokens"
    )
    print(f"Total prompt     {number(current.get('prompt_tokens_total')):>14} tokens")
    print(f"Requests running {number(current.get('num_requests_running')):>14}")
    print(f"Requests waiting {number(current.get('num_requests_waiting')):>14}")
    cache_percent = cache * 100 if cache is not None else None
    acceptance_percent = acceptance * 100 if acceptance is not None else None
    mean_length = 1 + accepted_per_step if accepted_per_step is not None else None
    print(f"KV cache usage   {number(cache_percent, 1):>14} %")
    print(f"MTP acceptance   {number(acceptance_percent, 1):>14} %")
    print(f"MTP mean length  {number(mean_length, 2):>14} tokens/step")
    print(f"Request errors   {number(current.get('request_success_total')):>14}")
    print("\nRates cover the last refresh interval; totals/MTP cover server uptime.")
    print("KV usage is the maximum across engines. Ctrl+C exits.")
    if previous is None:
        print(
            "Server counters reset." if reset else "Collecting the first rate sample…"
        )
    print(flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default="http://127.0.0.1:8001", help="Server base URL"
    )
    parser.add_argument(
        "--interval", type=float, default=2, help="Refresh seconds (default: 2)"
    )
    parser.add_argument(
        "--count", type=int, default=0, help="Number of refreshes; 0 runs until Ctrl+C"
    )
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0 or args.count < 0:
        parser.error(
            "--interval must be positive and finite; --count must be nonnegative"
        )
    url = args.url.rstrip("/")
    if not url.endswith("/metrics"):
        url += "/metrics"
    previous = None
    previous_time = time.monotonic()
    try:
        iteration = 0
        while args.count == 0 or iteration < args.count:
            try:
                current = fetch(url)
                now = time.monotonic()
                display(url, current, previous, now - previous_time)
                previous, previous_time = current, now
            except (URLError, OSError, ValueError) as error:
                print(f"Cannot read {url}: {error}. Retrying next refresh.", flush=True)
                previous = None
            iteration += 1
            if args.count == 0 or iteration < args.count:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
