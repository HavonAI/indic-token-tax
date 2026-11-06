"""95% bootstrap confidence intervals for Paper 2's FLORES-200 ratios (candidate tokens / rival tokens).

Per language, sentences are resampled with replacement 1,000 times (seed 42, paired: the same sentences for both
tokenizers), giving a 95% interval for the ratio. A win is "significant" when the whole interval is below 1.

Usage: python -m tokenizer_v3.bootstrap_ci --flores flores200_dataset.tar.gz --split devtest \
          --candidate "v4-200k=tokenizers/havon_v4/tokenizer_200k.json" --rival "GPT-4o (o200k_base)" \
          --rival "Sarvam-30B/105B" --out results/v4/ci
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .eval_world import RIVALS, load_flores

INDIA = ["hin_Deva", "npi_Deva", "urd_Arab", "awa_Deva", "snd_Arab", "mag_Deva", "ben_Beng", "hne_Deva", "mai_Deva",
         "bho_Deva", "guj_Gujr", "lus_Latn", "mar_Deva", "tel_Telu", "mal_Mlym", "tam_Taml", "kan_Knda", "asm_Beng",
         "san_Deva", "kas_Deva", "kas_Arab", "pan_Guru", "mni_Beng", "ory_Orya", "sat_Olck"]


def per_sentence(kind: str, source: str, revision: str | None):
    """Token count of each sentence (not the total), so sentences can be resampled."""
    if kind == "tiktoken":
        import tiktoken
        enc = tiktoken.get_encoding(source)
        return lambda texts: np.array([len(x) for x in enc.encode_ordinary_batch(texts)])
    from tokenizers import Tokenizer
    if kind == "hf":
        from huggingface_hub import hf_hub_download
        tok = Tokenizer.from_file(hf_hub_download(source, "tokenizer.json", revision=revision))
    else:
        tok = Tokenizer.from_file(source)
    return lambda texts: np.array([len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False)])


def ci(a: np.ndarray, b: np.ndarray, n: int, rng: np.random.Generator) -> tuple[float, float, float]:
    idx = rng.integers(0, len(a), size=(n, len(a)))
    r = a[idx].sum(1) / b[idx].sum(1)
    return float(a.sum() / b.sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--flores", type=Path, required=True)
    p.add_argument("--split", default="devtest")
    p.add_argument("--candidate", action="append", required=True, help="NAME=path/to/tokenizer.json")
    p.add_argument("--rival", action="append", required=True, help="a name from eval_world.RIVALS")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    flores = load_flores(a.flores, a.split)
    langs = sorted(flores)
    counts = {}
    for item in a.candidate:
        name, path = item.split("=", 1)
        fn = per_sentence("local", path, None)
        counts[name] = {l: fn(flores[l]) for l in langs}
    for r in a.rival:
        fn = per_sentence(*RIVALS[r])
        counts[r] = {l: fn(flores[l]) for l in langs}
    rows, summary = [], {"split": a.split, "bootstrap": a.n, "seed": 42, "pairs": {}}
    for item in a.candidate:
        cand = item.split("=", 1)[0]
        for r in a.rival:
            rng = np.random.default_rng(42)
            res = {l: ci(counts[cand][l], counts[r][l], a.n, rng) for l in langs}
            for l, (pt, lo, hi) in res.items():
                rows.append([cand, r, l, round(pt, 4), round(lo, 4), round(hi, 4)])
            sig = [l for l in langs if res[l][2] < 1]
            sig_loss = [l for l in langs if res[l][1] > 1]
            summary["pairs"][f"{cand} vs {r}"] = {
                "wins_point": sum(res[l][0] < 1 for l in langs), "wins_significant": len(sig),
                "losses_significant": len(sig_loss), "unclear": len(langs) - len(sig) - len(sig_loss),
                "india_wins_significant": sum(l in sig for l in INDIA),
                "english": [round(x, 4) for x in res["eng_Latn"]],
                "widest_interval": round(max(hi - lo for _, lo, hi in res.values()), 4)}
    with open(a.out / f"ci_{a.split}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate", "rival", "lang", "ratio", "ci_low", "ci_high"])
        w.writerows(rows)
    (a.out / f"ci_summary_{a.split}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
