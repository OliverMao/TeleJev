#!/usr/bin/env python3
"""Compare direct option readout against full autoregressive generation.

Both methods answer the same multi-task question about one image:

- direct: shared-prefix batch (image prefill once + one batched criteria forward)
- generation: a single greedy ``model.generate`` call that emits the whole
  answer sequence token by token (no manual decoding loop)

Run on a single visible CUDA GPU, e.g.::

    CUDA_VISIBLE_DEVICES=0 python examples/compare_generation.py \
      --model Qwen/Qwen3.5-4B \
      --image examples/fall.png
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telejev.autoregressive import generate_answers  # noqa: E402
from telejev.batch import score_batch  # noqa: E402
from telejev.core import load_causal_model  # noqa: E402

TASKS = [
    ("fight", "打架", "正在打架"),
    ("fall", "摔倒", "正在摔倒或已经摔倒"),
    ("wave", "挥手", "正在挥手"),
    ("chest", "捂胸口", "正在捂胸口"),
]


def build_criteria() -> list[dict]:
    return [
        {
            "id": key,
            "label": label,
            "question": f"画面中是否有人{verb}？",
            "options": [
                {"id": "yes", "description": f"画面中有人{verb}。"},
                {"id": "no", "description": f"画面中没有人{verb}。"},
            ],
        }
        for key, label, verb in TASKS
    ]


def direct_labels(results: list[dict]) -> list[str]:
    labels = []
    for result in results:
        probabilities = result["probabilities"]
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        labels.append(result["option_ids"][best])
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--image", default="examples/fall.png", help="Image path/URL/data URI ('' to skip)")
    parser.add_argument("--state", default="监控画面截图。")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Direct-mode input token limit")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Generation cap (default: 8*N+16)")
    args = parser.parse_args()

    model, tokenizer, processor, metadata = load_causal_model(args.model)
    print(f"model: {args.model}  multimodal={metadata['multimodal']}  image={args.image or '(none)'}\n")

    criteria = build_criteria()

    results, direct_timing = score_batch(
        model, tokenizer, metadata, args.state, args.image or None, criteria,
        args.max_tokens, processor,
    )
    direct = direct_labels(results)

    generation = generate_answers(
        model, tokenizer, processor, args.state, criteria,
        args.image or None, args.max_new_tokens,
    )
    generated = list(generation["answers"]) + ["?"] * (len(criteria) - len(generation["answers"]))

    print(f"{'task':<10}{'direct':<10}{'generate':<10}{'agree':<7}")
    agree = 0
    for (key, label, _), direct_label, generated_label in zip(TASKS, direct, generated):
        same = direct_label == generated_label
        agree += same
        print(f"{label:<10}{direct_label:<10}{generated_label:<10}{'yes' if same else 'no':<7}")
    print(f"\nagreement: {agree}/{len(criteria)}")
    print(f"generated text: {generation['text']!r}\n")

    print("direct (shared-prefix batch):")
    print(f"  image prefill : {direct_timing['prefill_seconds']:.4f} s")
    print(f"  criteria fwd  : {direct_timing['suffix_seconds']:.4f} s")
    print(f"  total         : {direct_timing['inference_seconds']:.4f} s  ({direct_timing['batch_size']} tasks, 2 forwards)")

    print("autoregressive (model.generate, greedy):")
    print(f"  prompt tokens : {generation['prompt_tokens']}")
    print(f"  new tokens    : {generation['new_tokens']}")
    print(f"  generate      : {generation['generate_seconds']:.4f} s")
    if generation["tokens_per_second"]:
        print(f"  throughput    : {generation['tokens_per_second']:.1f} tok/s")
    print(f"  total         : {generation['total_seconds']:.4f} s")

    if direct_timing["inference_seconds"] > 0 and generation["generate_seconds"] > 0:
        print(f"\nspeedup (generate / direct): {generation['generate_seconds'] / direct_timing['inference_seconds']:.1f}x")
        print("(reads one letter per task vs emitting a whole answer sequence)")


if __name__ == "__main__":
    main()