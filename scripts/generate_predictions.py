#!/usr/bin/env python3
"""Minimal local / API inference over a ROLL test jsonl."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def gen_api(prompt: str, *, model: str, base_url: str, api_key: str, temperature: float) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url or None)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


def gen_local(prompt: str, *, model_path: str, temperature: float, max_new_tokens: int) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=0.95,
    )
    gen = out[0][inputs["input_ids"].shape[-1] :]
    return tok.decode(gen, skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll_test", required=True)
    ap.add_argument("--out_pred", required=True)
    ap.add_argument("--llm_mode", choices=["api", "local"], default="api")
    ap.add_argument("--api_model", default="gpt-4o-mini")
    ap.add_argument("--base_url", default="")
    ap.add_argument("--api_key", default="")
    ap.add_argument("--local_model_path", default="")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = read_jsonl(Path(args.roll_test))
    if args.limit > 0:
        rows = rows[: args.limit]

    preds = []
    for i, row in enumerate(rows):
        prompt = row.get("prompt") or ""
        print(f"[{i+1}/{len(rows)}] generating {row.get('id')} ...")
        if args.llm_mode == "api":
            text = gen_api(
                prompt,
                model=args.api_model,
                base_url=args.base_url,
                api_key=args.api_key,
                temperature=args.temperature,
            )
        else:
            if not args.local_model_path:
                raise SystemExit("--local_model_path is required for local mode")
            text = gen_local(
                prompt,
                model_path=args.local_model_path,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
        preds.append({"id": row.get("id"), "prediction": text})

    write_jsonl(Path(args.out_pred), preds)
    print(f"Wrote {len(preds)} predictions -> {args.out_pred}")


if __name__ == "__main__":
    main()
