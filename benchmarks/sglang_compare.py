#!/usr/bin/env python3
"""Compare Jev-style readout and generation, both served by SGLang.

Start an SGLang server first, e.g.::

    python -m sglang.launch_server --model-path /path/to/model --port 30000

Then::

    python benchmarks/sglang_compare.py --base-url http://127.0.0.1:30000 --model Qwen/Qwen3.5-4B --image demo/fall.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telejev.sglang_backend import SGLangBackend  # noqa: E402

TASKS = [
    ("person", "有人", "画面中是否有人？", "画面中有人。", "画面中没有人。"),
    ("fight", "打架", "画面中是否有人正在打架？", "画面中有人正在打架。", "画面中没有人正在打架。"),
    ("fall", "摔倒", "画面中是否有人正在摔倒或已经摔倒？", "画面中有人正在摔倒。", "画面中没有人正在摔倒。"),
    ("wave", "挥手", "画面中是否有人正在挥手？", "画面中有人正在挥手。", "画面中没有人正在挥手。"),
    ("chest", "捂胸口", "画面中是否有人正在捂胸口？", "画面中有人正在捂胸口。", "画面中没有人正在捂胸口。"),
]


def build_criteria() -> list[dict]:
    return [
        {
            "id": key,
            "label": label,
            "question": question,
            "options": [{"id": "yes", "description": yes}, {"id": "no", "description": no}],
        }
        for key, label, question, yes, no in TASKS
    ]


def labels_by_id(results: list[dict]) -> list[str]:
    labels = []
    for result in results:
        probabilities = result["probabilities"]
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        labels.append(result["option_ids"][best])
    return labels


def to_output(labels: list[str]) -> dict:
    return {
        "has_person": 1 if labels and labels[0] == "yes" else 0,
        "violations": [TASKS[i][1] for i in range(1, len(TASKS)) if labels[i] == "yes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", default="demo/fall.png")
    parser.add_argument("--state", default="监控画面截图。")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--backend-name", default="sglang", help="label only: sglang or vllm")
    args = parser.parse_args()

    backend = SGLangBackend(args.base_url, args.model, backend_name=args.backend_name)
    criteria = build_criteria()

    results, direct_timing = backend.score_batch(args.state, args.image or None, criteria)
    direct = labels_by_id(results)

    generation = backend.generate(args.state, args.image or None, criteria, args.max_new_tokens)
    generated = list(generation["answers"]) + ["?"] * (len(criteria) - len(generation["answers"]))

    print(f"{'criterion':<12}{'Jev':<8}{'generate':<10}{'agree':<7}")
    agree = 0
    for (key, label, *_rest), direct_label, gen_label in zip(TASKS, direct, generated):
        same = direct_label == gen_label
        agree += same
        print(f"{label:<12}{direct_label:<8}{gen_label:<10}{'yes' if same else 'no':<7}")
    print(f"\nagreement: {agree}/{len(criteria)}")

    print("\nJev (prefill-only readout, concurrent batch):")
    print(f"  requests      : {direct_timing['requests']} (concurrency {direct_timing.get('concurrency')})")
    print(f"  if sequential : {direct_timing.get('sum_request_seconds', 0):.4f} s")
    print(f"  batch wall    : {direct_timing['total_seconds']:.4f} s")
    print(f"  output        : {to_output(direct)}")

    print("\ngenerate (same server):")
    print(f"  new tokens    : {generation['new_tokens']}")
    print(f"  generate      : {generation['generate_seconds']:.4f} s")
    print(f"  total         : {generation['total_seconds']:.4f} s")
    print(f"  text          : {generation['text']!r}")
    print(f"  output        : {generation.get('output')}")

    if direct_timing["total_seconds"] > 0 and generation["generate_seconds"] > 0:
        print(f"\ngenerate / Jev: {generation['generate_seconds'] / direct_timing['total_seconds']:.1f}x")


if __name__ == "__main__":
    main()