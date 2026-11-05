"""Train Havon tokenizer v3 (multilingual, 256k vocabulary) on a Kaggle CPU session.

Phase 1: sample text per language, each language in its own subprocess with retries, cached to disk,
         so a native crash in a streaming library costs one language attempt, not the whole run.
Phase 2: train one byte-level BPE at the largest vocabulary. Smaller vocabularies are exact prefixes
         of it (BPE learns merges in order), so they are cut from this one run by truncate.py.

Usage: python -m tokenizer_v3.train_v3 --out /kaggle/working [--vocab 262144] [--scale 1.0]
       --scale shrinks every budget (e.g. 0.001 for a quick local test). --only a,b limits languages.
"""
import argparse
import itertools
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("tok_v3")
MIN_CHARS = 100
# Same letter+mark rule as Havon v2 and single digits, plus chunking for scripts written without spaces:
# Han and Japanese kana in runs of at most 3 characters; Thai, Lao, Khmer, Myanmar and Tibetan in runs of at
# most 6 letters, each with its marks attached. Without this, whole sentences become single pre-tokens and
# BPE training ran past 32 GB. The general letter rule excludes these scripts, so it cannot swallow them
# (an optional leading space or punctuation mark is allowed on every rule, as in GPT-style patterns).
NO_SPACE = r"\p{Han}\p{Hiragana}\p{Katakana}\p{Thai}\p{Lao}\p{Khmer}\p{Myanmar}\p{Tibetan}"
SPLIT_PATTERN = (r"[^\r\n\p{L}\p{M}\p{N}]?[\p{Han}\p{Hiragana}\p{Katakana}]{1,3}"
                 r"|[^\r\n\p{L}\p{M}\p{N}]?(?:[[\p{Thai}\p{Lao}\p{Khmer}\p{Myanmar}\p{Tibetan}]&&\P{M}]\p{M}*){1,6}"
                 r"|[^\r\n\p{L}\p{M}\p{N}]?[[\p{L}\p{M}]&&[^" + NO_SPACE + r"]]+"
                 r"|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
SPECIAL_TOKENS = ["<|eos|>", "<|pad|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end_turn|>",
                  "<|think|>", "<|/think|>", "<|tool_call|>", "<|/tool_call|>", "<|tool_result|>",
                  "<|/tool_result|>"] + [f"<|reserved_{i}|>" for i in range(52)]
FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"


def parquet_texts(pattern: str):
    """Read only the `text` column straight from the Hub's parquet files.

    Avoids the datasets library's strict schema cast (some FineWeb-2 files carry an extra column) and makes
    far fewer API calls than streaming load_dataset, which rate-limited unauthenticated parallel workers.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    files = sorted(fs.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no parquet files match {pattern}")
    for path in files:
        with fs.open(path, "rb", block_size=8 << 20) as fh:
            for batch in pq.ParquetFile(fh).iter_batches(columns=["text"], batch_size=512):
                for text in batch.column(0).to_pylist():
                    if text and len(text) >= MIN_CHARS:
                        yield text


def fw2_texts(config: str):
    return parquet_texts(f"datasets/HuggingFaceFW/fineweb-2/data/{config}/train/*.parquet")


def sangraha_texts(folder: str):
    return parquet_texts(f"datasets/ai4bharat/sangraha/verified/{folder}/*.parquet")


def special_texts(name: str):
    """English, romanized, maths and code reuse Havon's existing human-written sources."""
    from havonlm import sources
    lang = {"eng": "eng", "hin_rom": "hin_rom", "mar_rom": "mar_rom", "math": "math", "code": "code"}[name]
    return sources.stream(lang)


def texts_for(name: str, spec: dict):
    if spec["kind"] == "special":
        return special_texts(name)
    if spec["kind"] == "indic":
        streams = [sangraha_texts(spec["sangraha"])]
        if spec.get("fw2"):
            streams.append(fw2_texts(spec["fw2"]))
        return (t for pair in itertools.zip_longest(*streams) for t in pair if t)
    return fw2_texts(spec["fw2"])


def sample_one(name: str, spec: dict, folder: Path) -> None:
    """Worker body: write up to spec['chars'] characters, then rename .part to .txt atomically."""
    tmp, seen = folder / f"{name}.part", 0
    with open(tmp, "w", encoding="utf-8") as f:
        for text in texts_for(name, spec):
            f.write(text.replace("\x00", "") + "\x00")
            seen += len(text)
            if seen >= spec["chars"]:
                break
    tmp.replace(folder / f"{name}.txt")
    print(f"SAMPLED {name} {seen}", flush=True)
    os._exit(0)  # skip interpreter teardown: streaming threads have crashed at exit before


def sample_language(name: str, spec: dict, folder: Path, attempts: int = 5) -> int | None:
    out = folder / f"{name}.txt"
    for attempt in range(1, attempts + 1):
        if out.exists():
            break
        if attempt > 1:
            time.sleep(30 * 2 ** (attempt - 2))  # back off: 30s, 60s, 120s, 240s (Hub rate limits)
        try:
            rc = subprocess.run([sys.executable, "-X", "faulthandler", "-m", "tokenizer_v3.train_v3", "--worker",
                                 name, "--worker-spec", json.dumps(spec), "--out", str(folder)],
                                timeout=3600).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        log.info("%s attempt %d exit %s", name, attempt, rc)
    if not out.exists():
        log.warning("giving up on %s", name)
        return None
    return out.stat().st_size


def sample_all(plan: dict, folder: Path, workers: int = 4) -> dict[str, int]:
    """Languages are network-bound (dataset lookup dominates), so sample several at once."""
    from concurrent.futures import ThreadPoolExecutor
    folder.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        sizes = dict(zip(plan, pool.map(lambda kv: sample_language(kv[0], kv[1], folder), plan.items())))
    return {k: v for k, v in sizes.items() if v is not None}


def log_memory_every(seconds: int = 60) -> None:
    """Background thread: log this process's RSS and the machine's available memory, to diagnose OOM kills."""
    import threading

    def loop():
        while True:
            try:
                rss = int(next(l for l in open("/proc/self/status") if l.startswith("VmRSS")).split()[1]) / 1e6
                avail = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) / 1e6
                log.info("memory: rss %.1f GB, available %.1f GB", rss, avail)
            except (OSError, StopIteration):
                return  # not Linux; nothing to report
            time.sleep(seconds)

    threading.Thread(target=loop, daemon=True).start()


def train(files: list[Path], vocab: int, out: Path, fraction: float = 1.0) -> None:
    """Train on the first `fraction` of each language sample (lower it to fit memory without resampling)."""
    from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=SPECIAL_TOKENS, min_frequency=2, max_token_length=32,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)

    def corpus():
        for path in files:
            text = path.read_text(encoding="utf-8")
            budget, seen = int(len(text) * fraction), 0
            for doc in text.split("\x00"):
                if doc:
                    yield doc
                    seen += len(doc)
                    if seen >= budget:
                        break
            log.info("fed %s (%.1fM chars)", path.stem, seen / 1e6)

    log_memory_every()
    started = time.time()
    tok.train_from_iterator(corpus(), trainer=trainer)
    tok.save(str(out / f"havon_v3_{vocab // 1024}k.json"))
    log.info("trained vocab %d in %.0f min", tok.get_vocab_size(), (time.time() - started) / 60)


def flores_codes(cache: Path) -> list[str]:
    import tarfile
    import urllib.request
    tar_path = cache / "flores200_dataset.tar.gz"
    if not tar_path.exists():
        urllib.request.urlretrieve(FLORES_URL, tar_path)
    with tarfile.open(tar_path) as tar:
        return sorted(m.name.split("/")[-1][:-4] for m in tar if m.name.endswith(".dev"))


def fineweb2_configs() -> set[str]:
    import urllib.request
    with urllib.request.urlopen("https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-2") as r:
        return {c["config_name"] for c in json.load(r)["cardData"]["configs"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--vocab", type=int, default=262_144)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--only", default="")
    p.add_argument("--phase", choices=["all", "sample", "train"], default="all")
    p.add_argument("--samples", type=Path, default=None, help="train phase: folder searched for samples/*.txt")
    p.add_argument("--fraction", type=float, default=1.0, help="train phase: share of each sample to use")
    p.add_argument("--worker")
    p.add_argument("--worker-spec")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.worker:
        sample_one(args.worker, json.loads(args.worker_spec), args.out)
        return
    if args.phase == "train":
        files = sorted(p for p in (args.samples or args.out).rglob("*.txt") if p.parent.name == "samples")
        if not files:
            raise FileNotFoundError(f"no samples/*.txt under {args.samples or args.out}")
        log.info("training on %d samples, %.0f MB, fraction %.2f", len(files),
                 sum(f.stat().st_size for f in files) / 1e6, args.fraction)
        args.out.mkdir(parents=True, exist_ok=True)
        train(files, args.vocab, args.out, args.fraction)
        return
    from tokenizer_v3.langs import build_plan
    args.out.mkdir(parents=True, exist_ok=True)
    plan = build_plan(flores_codes(args.out), fineweb2_configs())
    if args.only:
        plan = {k: v for k, v in plan.items() if k in set(args.only.split(","))}
    plan = {k: {**v, "chars": max(1000, int(v["chars"] * args.scale))} for k, v in plan.items()}
    log.info("plan: %d samples, %.0fM chars", len(plan), sum(v["chars"] for v in plan.values()) / 1e6)
    got = sample_all(plan, args.out / "samples")
    (args.out / "sample_sizes.json").write_text(json.dumps(got, indent=1))
    missing = sorted(set(plan) - set(got))
    log.info("sampled %d of %d; missing: %s", len(got), len(plan), missing)
    if args.phase == "sample":
        return
    files = [args.out / "samples" / f"{k}.txt" for k in plan if k in got]
    train(files, args.vocab, args.out, args.fraction)


if __name__ == "__main__":
    main()
