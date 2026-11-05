"""SuperBPE stage 2 on top of Havon v3: learn multi-word tokens (Liu et al., 2025), trained from scratch on our text.

Starts from the first --inherit merges of Havon v3 (stage 1, normal BPE), then resumes training to --vocab
with a pre-tokenizer that no longer splits on spaces, so tokens may span words. Scripts written without spaces
keep v3's chunking, so their runs cannot become huge pre-tokens. Smaller vocabularies are cut from the result
with truncate.py (merges are learned in order).

Requires the SuperBPE fork of the tokenizers library (it resumes from merges.txt in the working directory).
Usage: python -m tokenizer_v3.superbpe_stage2 --samples DIR --base havon_v3_256k.json --out OUT [--fraction 0.3]
"""
import argparse
import json
import logging
import os
import threading
import time
from pathlib import Path


log = logging.getLogger("superbpe")
# Stage 2: v3's chunking for no-space scripts, single digits and punctuation runs are still split off;
# everything else (letters and the spaces between words) may merge into multi-word tokens.
STAGE2 = (r"[^\r\n\p{L}\p{M}\p{N}]?[\p{Han}\p{Hiragana}\p{Katakana}]{1,3}"
          r"|[^\r\n\p{L}\p{M}\p{N}]?(?:[[\p{Thai}\p{Lao}\p{Khmer}\p{Myanmar}\p{Tibetan}]&&\P{M}]\p{M}*){1,6}"
          r"|\p{N}| ?[^\s\p{L}\p{M}\p{N}]{2,}[\r\n/]*| +(?!\S)")


def log_memory(seconds: int = 60) -> None:
    def loop():
        while True:
            try:
                rss = int(next(l for l in open("/proc/self/status") if l.startswith("VmRSS")).split()[1]) / 1e6
                avail = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) / 1e6
                log.info("memory: rss %.1f GB, available %.1f GB", rss, avail)
            except (OSError, StopIteration):
                return
            time.sleep(seconds)
    threading.Thread(target=loop, daemon=True).start()


def write_corpus(samples: list[Path], fraction: float, folder: Path) -> list[str]:
    """One file per language: the first `fraction` of its sample, one document per line group."""
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for path in samples:
        text = path.read_text(encoding="utf-8")
        budget, seen, docs = int(len(text) * fraction), 0, []
        for doc in text.split("\x00"):
            if doc:
                docs.append(doc)
                seen += len(doc)
                if seen >= budget:
                    break
        dst = folder / path.name
        dst.write_text("\n".join(docs), encoding="utf-8")
        files.append(str(dst.resolve()))
    log.info("corpus: %d files, %.0f MB", len(files), sum(os.path.getsize(f) for f in files) / 1e6)
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True, help="searched for samples/*.txt")
    ap.add_argument("--base", type=Path, required=True, help="Havon v3 tokenizer.json (stage 1)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--inherit", type=int, default=210_000)
    ap.add_argument("--vocab", type=int, default=524_288)
    ap.add_argument("--fraction", type=float, default=0.3)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    samples = sorted(p for p in args.samples.rglob("*.txt") if p.parent.name == "samples")
    files = write_corpus(samples, args.fraction, args.out / "corpus")
    base = json.loads(args.base.read_text(encoding="utf-8"))
    merges = base["model"]["merges"]
    work = args.out / "stage2"
    work.mkdir(parents=True, exist_ok=True)
    lines = ["#version: 0.2"] + [" ".join(m) if isinstance(m, list) else m for m in merges[: args.inherit]]
    (work / "merges.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("inheriting %d of v3's %d merges; training to %d", args.inherit, len(merges), args.vocab)

    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(STAGE2), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    # Keep the base tokenizer's special tokens (v2 has 32, v3 has 64) so ids and chat markers line up.
    specials = [t["content"] for t in sorted(base.get("added_tokens", []), key=lambda t: t["id"]) if t.get("special")]
    trainer = trainers.BpeTrainer(vocab_size=args.vocab, special_tokens=specials, min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    log_memory()
    started = time.time()
    here = os.getcwd()
    os.chdir(work)  # the fork resumes from ./merges.txt
    try:
        tok.train(files, trainer)
    finally:
        os.chdir(here)
    out = args.out / f"havon_v4_superbpe_{args.vocab // 1024}k.json"
    tok.save(str(out))
    log.info("trained vocab %d in %.0f min -> %s", tok.get_vocab_size(), (time.time() - started) / 60, out.name)
    if args.vocab > 262_144:
        from tokenizer_v3.truncate import truncate
        truncate(out, 262_144, args.out / "havon_v4_superbpe_256k.json")
        log.info("cut 262,144-token prefix")


if __name__ == "__main__":
    main()
