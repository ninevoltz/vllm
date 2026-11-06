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
from PIL import Image
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
    args = parser.parse_args()

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
        assert record["passed"], record

    chat("text", "What is 2 + 2? Reply with only the number.", "4")
    png = io.BytesIO()
    Image.new("RGB", (224, 224), "red").save(png, format="PNG")
    chat(
        "vision",
        [
            {
                "type": "image_url",
                "image_url": {"url": data_url(png.getvalue(), "image/png")},
            },
            {
                "type": "text",
                "text": "What color fills the image? Reply with one word.",
            },
        ],
        "red",
    )
    video = io.BytesIO()
    with av.open(video, "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=2)
        stream.width = stream.height = 224
        stream.pix_fmt = "yuv420p"
        for color in ("red", "red", "blue", "blue"):
            pixels = np.asarray(Image.new("RGB", (224, 224), color))
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
                "text": "What color fills the final frames? Reply with one word.",
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
                "content": "Remember the secret code 71391.\n"
                + marker
                + "\nWhat is the secret code? Reply only with the code.",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    before, after = full.split(marker)
    prefix = tokenizer.encode(before, add_special_tokens=False)
    suffix = tokenizer.encode(after, add_special_tokens=False)
    filler = tokenizer.encode(" tree", add_special_tokens=False)
    assert len(filler) == 1, filler
    for length in args.lengths:
        # Reserve exactly 32 positions for generation, including EOS.
        prompt_len = length - 32
        prompt = prefix + filler * (prompt_len - len(prefix) - len(suffix)) + suffix
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
            },
        )
        answer = result["choices"][0]["text"]
        record = {
            "test": "context",
            "context_budget": length,
            "answer": answer,
            "usage": result["usage"],
            "seconds": time.monotonic() - started,
            "passed": "71391" in answer,
        }
        print(json.dumps(record), flush=True)
        assert result["usage"]["prompt_tokens"] == prompt_len, record
        assert record["passed"], record


if __name__ == "__main__":
    main()
