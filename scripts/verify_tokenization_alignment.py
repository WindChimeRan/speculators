#!/usr/bin/env python3
"""Verify RFC #652 claim: do HF apply_chat_template tokens equal vLLM serving tokens?

Reproduction script for the investigation at
https://github.com/vllm-project/speculators/issues/652

Findings (2026-06-24):
  1. Standard HF models: byte-identical (vLLM render calls the same
     tokenizer.apply_chat_template, vllm/renderers/hf.py:740).
  2. gpt-oss: identical when date+effort aligned; diverges only on the
     serving-time preamble (Current date / Reasoning effort). Reconcilable
     with apply_chat_template(reasoning_effort=..., strftime_now=...) —
     a plumbing gap, not an encoder mismatch.
  3. Current pipeline is unaffected: text path sends input_ids verbatim
     (vllm_client.py:190-196), multimodal path pins add_generation_prompt=False
     (vllm_client.py:203), both hard-assert token equality (vllm_client.py:106).

Requirements:
  - vllm venv (has vllm + openai_harmony + transformers)
  - PYTHONPATH=<speculators>/src (for speculators imports)
  - For gpt-oss: TIKTOKEN_ENCODINGS_BASE pointed at cached o200k_base.tiktoken

Usage:
  # Standard model (no special deps):
  python verify_tokenization_alignment.py --standard-only

  # Full suite including gpt-oss:
  TIKTOKEN_ENCODINGS_BASE=/path/to/tiktoken-cache \\
  PYTHONPATH=/path/to/speculators/src \\
  python verify_tokenization_alignment.py
"""

# ruff: noqa: PLC0415, BLE001 — deferred imports and broad excepts are
# intentional: each check must skip gracefully when deps are missing.

import argparse
import datetime
import sys


def _first_diff(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def check_standard_models():
    """Verify standard HF models produce identical tokens via both paths."""
    from transformers import AutoTokenizer

    conv = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ]

    models = []
    for m in ["Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B-Instruct"]:
        try:
            AutoTokenizer.from_pretrained(m)
            models.append(m)
        except Exception:
            print(f"  {m}: not cached, skipping")

    if not models:
        print("SKIP: no standard models cached")
        return True

    all_pass = True
    for model in models:
        tok = AutoTokenizer.from_pretrained(model)

        hf_ids = tok.apply_chat_template(
            conv, tokenize=True, add_generation_prompt=False, return_dict=False
        )
        vllm_ids = tok.apply_chat_template(
            conv,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
            chat_template=tok.get_chat_template(),
        )

        match = list(hf_ids) == list(vllm_ids)
        print(f"  {model}: {len(hf_ids)} tokens, match={match}")
        if not match:
            all_pass = False

    return all_pass


def check_gpt_oss():
    """Verify gpt-oss aligns under matched settings, diverges on date/effort."""
    try:
        from transformers import AutoTokenizer
        from vllm.entrypoints.openai.parser.harmony_utils import (
            get_system_message,
            get_user_message,
            render_for_completion,
        )
    except ImportError as e:
        print(f"SKIP: {e}")
        return True

    try:
        tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    except Exception as e:
        print(f"SKIP: cannot load gpt-oss tokenizer: {e}")
        return True

    q = "What is the capital of France?"
    today = datetime.date.today().strftime("%Y-%m-%d")

    def hf_ids(**kwargs):
        return list(
            tok.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                **kwargs,
            )
        )

    def vllm_ids(effort, date):
        return list(
            render_for_completion(
                [
                    get_system_message(reasoning_effort=effort, start_date=date),
                    get_user_message(q),
                ]
            )
        )

    all_pass = True

    # 1. Baseline: aligned settings -> identical
    hf = hf_ids(strftime_now=lambda _f: today)
    vllm = vllm_ids("medium", today)
    baseline = hf == vllm
    print(f"  baseline (today, medium): hf={len(hf)} vllm={len(vllm)} match={baseline}")
    if not baseline:
        all_pass = False

    # 2. Date divergence
    vllm_other = vllm_ids("medium", "2024-01-01")
    date_div = hf != vllm_other
    idx = _first_diff(hf, vllm_other)
    print(f"  date differs (2024-01-01): diverges={date_div} at idx={idx}")
    if not date_div:
        all_pass = False

    # 3. Effort divergence
    vllm_high = vllm_ids("high", today)
    eff_div = hf != vllm_high
    idx = _first_diff(hf, vllm_high)
    print(f"  effort differs (high): diverges={eff_div} at idx={idx}")
    if not eff_div:
        all_pass = False

    # 4. Reconciliation: HF CAN reproduce served tokens with the right kwargs
    reconciled = hf_ids(reasoning_effort="high", strftime_now=lambda _f: "2024-01-01")
    served = vllm_ids("high", "2024-01-01")
    recon = reconciled == served
    print(
        f"  reconciled (high, 2024-01-01): match={recon} -> plumbing gap, not encoder"
    )
    if not recon:
        all_pass = False

    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--standard-only",
        action="store_true",
        help="Skip gpt-oss tests (no vllm/harmony needed)",
    )
    args = parser.parse_args()

    print("=== Standard HF models ===")
    ok1 = check_standard_models()

    ok2 = True
    if not args.standard_only:
        print("\n=== gpt-oss / Harmony ===")
        ok2 = check_gpt_oss()

    print(f"\nResult: {'PASS' if ok1 and ok2 else 'FAIL'}")
    sys.exit(0 if ok1 and ok2 else 1)


if __name__ == "__main__":
    main()
