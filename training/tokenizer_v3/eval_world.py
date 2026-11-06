"""Compare tokenizers on every FLORES-200 language: tokens needed for the same sentences.

Selection must use --split dev; report final numbers on --split devtest only (never tuned on).
Output: CSV of tokens per language per tokenizer, and a summary counting, for each candidate,
the languages where it needs fewer tokens than each rival.

Usage: python -m tokenizer_v3.eval_world --flores PATH/flores200_dataset.tar.gz --split dev \
         --candidate "Havon v3 128k=PATH.json" [--candidate ...] --out results_dir
"""
import argparse
import csv
import json
import tarfile
import unicodedata
from pathlib import Path

RIVALS = {  # name -> (kind, source, revision)
    "GPT-4o (o200k_base)": ("tiktoken", "o200k_base", None),
    "GPT-4 (cl100k_base)": ("tiktoken", "cl100k_base", None),
    "Sarvam-30B/105B": ("hf", "sarvamai/sarvam-30b", "071ae95"),
    "Sarvam-1": ("hf", "sarvamai/sarvam-1", "e960733"),
    "Qwen2.5": ("hf", "Qwen/Qwen2.5-7B-Instruct", "a09a354"),
    "DeepSeek-V3": ("hf", "deepseek-ai/DeepSeek-V3", "e815299"),
}


def load_flores(tar_path: Path, split: str) -> dict[str, list[str]]:
    out = {}
    with tarfile.open(tar_path) as tar:
        for m in tar:
            if m.isfile() and m.name.endswith(f".{split}") and f"/{split}/" in m.name:
                lang = m.name.split("/")[-1][: -len(split) - 1]
                lines = tar.extractfile(m).read().decode("utf-8").splitlines()
                out[lang] = [unicodedata.normalize("NFC", s) for s in lines]
    return out


def counter(kind: str, source: str, revision: str | None):
    if kind == "tiktoken":
        import tiktoken
        enc = tiktoken.get_encoding(source)
        return lambda texts: sum(len(x) for x in enc.encode_ordinary_batch(texts))
    from tokenizers import Tokenizer
    if kind == "hf":
        from huggingface_hub import hf_hub_download
        tok = Tokenizer.from_file(hf_hub_download(source, "tokenizer.json", revision=revision))
    else:
        tok = Tokenizer.from_file(source)
    return lambda texts: sum(len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--flores", type=Path, required=True)
    p.add_argument("--split", choices=["dev", "devtest"], required=True)
    p.add_argument("--candidate", action="append", default=[], help="NAME=path/to/tokenizer.json")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    flores = load_flores(args.flores, args.split)
    tokenizers = {name: counter(*spec) for name, spec in RIVALS.items()}
    candidates = []
    for item in args.candidate:
        name, path = item.split("=", 1)
        tokenizers[name] = counter("local", path, None)
        candidates.append(name)

    counts = {name: {lang: fn(texts) for lang, texts in flores.items()} for name, fn in tokenizers.items()}
    langs = sorted(flores)
    with open(args.out / f"world_{args.split}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lang"] + list(tokenizers))
        for lang in langs:
            w.writerow([lang] + [counts[t][lang] for t in tokenizers])

    gpt_eng = counts["GPT-4o (o200k_base)"]["eng_Latn"]
    summary = {"split": args.split, "languages": len(langs), "candidates": {}}
    for cand in candidates:
        rivals = {}
        for rival in RIVALS:
            wins = [l for l in langs if counts[cand][l] < counts[rival][l]]
            rivals[rival] = {"wins": len(wins), "losses": [l for l in langs if counts[cand][l] > counts[rival][l]]}
        summary["candidates"][cand] = {
            "english_vs_gpt4o": round(counts[cand]["eng_Latn"] / gpt_eng, 4),
            "beats": rivals,
        }
    (args.out / f"world_summary_{args.split}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for cand, s in summary["candidates"].items():
        print(cand, "English vs GPT-4o:", s["english_vs_gpt4o"],
              {r: f"{v['wins']}/{len(langs)}" for r, v in s["beats"].items()})


if __name__ == "__main__":
    main()
