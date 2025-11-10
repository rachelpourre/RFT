import json
import math
from pathlib import Path
from tqdm import tqdm

from .cot import CoTModel
from .data import Dataset


def _is_close(a: float, b: float, rel_tol: float = 1e-3, abs_tol: float = 1e-3) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    except Exception:
        return False


def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    model = CoTModel()
    dataset = Dataset("train")

    out = []
    total = 0
    succeeded = 0

    for q, gold in tqdm(dataset, desc="Generating RFT dataset"):
        prompt = model.format_prompt(q)
        try:
            gens_grouped = model.batched_generate([prompt], num_return_sequences=oversample, temperature=temperature)
            gens = gens_grouped[0] if isinstance(gens_grouped, list) and len(gens_grouped) > 0 else []
        except Exception as e:
            print(f"Generation failed for question: {q!r}  -- {e}")
            continue

        picked = None
        for g in gens:
            try:
                parsed = model.parse_answer(g)
            except Exception:
                parsed = float("nan")
            if _is_close(parsed, gold):
                picked = g
                break

        if picked is not None:
            out.append([q, float(gold), picked])
            succeeded += 1
        # If none matched, skip this question 

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(out)} examples to {out_path} (asked {total}, succeeded {succeeded}, rate={succeeded/total:.3f})")


if __name__ == "__main__":
    from fire import Fire

    Fire({"generate_dataset": generate_dataset, "main": generate_dataset})
