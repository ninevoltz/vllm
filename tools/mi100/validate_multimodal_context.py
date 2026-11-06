# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate a local MI100 server with images, video and exact token budgets."""

import argparse
import io
import json
import time
import urllib.request

import av
import numpy as np
import pybase64 as base64
from PIL import Image, ImageDraw
from transformers import AutoTokenizer


def request(base, route, body):
    req = urllib.request.Request(
        base + route,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=43200) as response:
        return json.load(response)


def data_url(data, mime):
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local checkpoint path")
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--lengths", nargs="+", type=int, default=[8192, 32768, 131072, 262144]
    )
    parser.add_argument("--context-only", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args()

    failures = []

    def chat(name, content, expected):
        started = time.monotonic()
        result = request(
            args.base,
            "/v1/chat/completions",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "max_tokens": 128,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        answer = result["choices"][0]["message"]["content"] or ""
        record = {
            "test": name,
            "answer": answer,
            "usage": result["usage"],
            "seconds": time.monotonic() - started,
            "passed": expected in answer.lower(),
        }
        print(json.dumps(record), flush=True)
        if not record["passed"]:
            failures.append(record)
            if not args.continue_on_failure:
                raise AssertionError(record)

    if not args.context_only:
        chat("text", "What is 2 + 2? Reply with only the number.", "4")
        png = io.BytesIO()
        scene = Image.new("RGB", (448, 448), "white")
        draw = ImageDraw.Draw(scene)
        draw.rectangle((40, 40, 200, 200), fill="red")
        draw.ellipse((240, 240, 410, 410), fill="blue")
        scene.save(png, format="PNG")
        chat(
            "vision",
            [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url(png.getvalue(), "image/png")},
                },
                {
                    "type": "text",
                    "text": "Which color is the upper-left square? One word.",
                },
            ],
            "red",
        )
        video = io.BytesIO()
        with av.open(video, "w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=2)
            stream.width = stream.height = 224
            stream.pix_fmt = "yuv420p"
            for color in ("red",) * 4 + ("blue",) * 4:
                scene = Image.new("RGB", (224, 224), "white")
                ImageDraw.Draw(scene).rectangle((40, 40, 180, 180), fill=color)
                pixels = np.asarray(scene)
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        chat(
            "video",
            [
                {
                    "type": "video_url",
                    "video_url": {"url": data_url(video.getvalue(), "video/mp4")},
                },
                {
                    "type": "text",
                    "text": "Which color is the last square? One word.",
                },
            ],
            "blue",
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    marker = "MI100_CONTEXT_FILLER"
    full = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    "Inventory record: warehouse W172 is located in Oslo.\n"
                    "Background notes:\n"
                )
                + marker
                + (
                    "\nWhich city is warehouse W172 located in? "
                    "Reply only with the city name."
                ),
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    before, after = full.split(marker)
    suffix = tokenizer.encode(after, add_special_tokens=False)
    cities = ("Riga", "Boston", "Delhi", "Accra", "Berlin", "Sofia")
    notes = "\n".join(
        f"Record X{i:06}: destination {cities[i % len(cities)]}; "
        f"shipment count {i % 97 + 1}; packaging cardboard."
        for i in range(30000)
    )
    filler = tokenizer.encode(
        notes, add_special_tokens=False, truncation=True, max_length=max(args.lengths)
    )
    for index, length in enumerate(args.lengths):
        city = ("Oslo", "Kyoto", "Lima", "Paris")[index % 4]
        prefix = tokenizer.encode(
            before.replace("Oslo", city), add_special_tokens=False
        )
        # Reserve exactly 32 positions for generation, including EOS.
        prompt_len = length - 32
        count = prompt_len - len(prefix) - len(suffix)
        assert 0 <= count <= len(filler)
        prompt = prefix + filler[:count] + suffix
        assert len(prompt) == prompt_len
        started = time.monotonic()
        result = request(
            args.base,
            "/v1/completions",
            {
                "model": args.model,
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": 32,
                "min_tokens": 32,
                "ignore_eos": True,
            },
        )
        answer = result["choices"][0]["text"]
        record = {
            "test": "context",
            "context_budget": length,
            "answer": answer,
            "usage": result["usage"],
            "seconds": time.monotonic() - started,
            "passed": city.lower() in answer.lower(),
        }
        print(json.dumps(record), flush=True)
        assert result["usage"]["prompt_tokens"] == prompt_len, record
        assert result["usage"]["completion_tokens"] == 32, record
        if not record["passed"]:
            failures.append(record)
            if not args.continue_on_failure:
                raise AssertionError(record)

    if failures:
        raise SystemExit(f"{len(failures)} validation checks failed")


if __name__ == "__main__":
    main()
