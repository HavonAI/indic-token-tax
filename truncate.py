"""Cut a smaller tokenizer from a larger BPE tokenizer.json.

BPE training adds merges one at a time in a fixed order, so the first K tokens (special tokens, byte
alphabet, then the earliest merges) are what training straight to vocabulary K would produce.

Usage: python truncate.py BIG.json 65536 OUT.json
"""
import json
import sys
from pathlib import Path


def truncate(src: Path, vocab_size: int, dst: Path) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))
    vocab: dict[str, int] = data["model"]["vocab"]
    if vocab_size >= len(vocab):
        raise ValueError(f"target {vocab_size} is not smaller than source vocab {len(vocab)}")
    keep = {tok: i for tok, i in vocab.items() if i < vocab_size}
    merges = data["model"]["merges"]

    def merged(m) -> str:
        a, b = m if isinstance(m, list) else m.split(" ", 1)
        return a + b

    data["model"]["vocab"] = keep
    data["model"]["merges"] = [m for m in merges if merged(m) in keep]
    data["added_tokens"] = [t for t in data.get("added_tokens", []) if t["id"] < vocab_size]
    dst.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    truncate(Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]))
