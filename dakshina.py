"""Romanized vs native-script token cost on the Dakshina dataset (Roark et al., LREC 2020, CC BY-SA 4.0).

Dakshina gives 10,000 Wikipedia strings per language, each romanized by native-speaker annotators, for
12 South Asian languages. For every tokenizer we report the parity of the romanized text against the
same strings in native script (tokens_roman / tokens_native, with a bootstrap 95% CI).

The 2 GB archive is streamed once; only the 12 `*.romanized.rejoined.tsv` files are kept, and the
archive's published MD5 is checked while streaming.

Usage: python dakshina.py [--tokenizer NAME=PATH ...]   (run after measure.py; reuses its pinned tokenizers)
"""
import argparse
import base64
import csv
import hashlib
import json
import tarfile
import unicodedata
import urllib.request
from pathlib import Path

import measure

URL = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"
MD5_B64 = "pWCGix1nAmAsTqToN5T6Jg=="  # x-goog-hash md5 published by Google Cloud Storage
LANGS = {"bn": "ben_Beng", "gu": "guj_Gujr", "hi": "hin_Deva", "kn": "kan_Knda", "ml": "mal_Mlym",
         "mr": "mar_Deva", "pa": "pan_Guru", "sd": "snd_Arab", "si": "sin_Sinh", "ta": "tam_Taml",
         "te": "tel_Telu", "ur": "urd_Arab"}
OUT_DIR = measure.DATA / "dakshina"


class HashingReader:
    def __init__(self, raw):
        self.raw, self.md5 = raw, hashlib.md5()

    def read(self, n=-1):
        chunk = self.raw.read(n)
        self.md5.update(chunk)
        return chunk


def fetch() -> None:
    wanted = {f"dakshina_dataset_v1.0/{l}/romanized/{l}.romanized.rejoined.tsv": l for l in LANGS}
    if all((OUT_DIR / f"{l}.tsv").exists() for l in LANGS):
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL) as resp:
        reader = HashingReader(resp)
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            for member in tar:
                lang = wanted.get(member.name.lstrip("./"))
                if lang and member.isfile():
                    (OUT_DIR / f"{lang}.tsv").write_bytes(tar.extractfile(member).read())
        while reader.read(1 << 20):  # finish the stream so the MD5 covers the whole archive
            pass
    got = base64.b64encode(reader.md5.digest()).decode()
    if got != MD5_B64:
        raise RuntimeError(f"Dakshina archive MD5 mismatch: {got} != {MD5_B64}")
    missing = [l for l in LANGS if not (OUT_DIR / f"{l}.tsv").exists()]
    if missing:
        raise RuntimeError(f"files not found in archive for: {missing}")


def load_pairs(lang: str) -> tuple[list[str], list[str]]:
    native, roman = [], []
    for line in (OUT_DIR / f"{lang}.tsv").read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
            native.append(unicodedata.normalize("NFC", parts[0].strip()))
            roman.append(unicodedata.normalize("NFC", parts[1].strip()))
    return native, roman


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", action="append", default=[], help="extra NAME=path/to/tokenizer.json")
    args = ap.parse_args()
    fetch()
    toks, meta = measure.load_tokenizers(args.tokenizer)
    rows = []
    for lang, flores in LANGS.items():
        native, roman = load_pairs(lang)
        for name, fn in toks.items():
            tn, tr = measure.counts(fn, native), measure.counts(fn, roman)
            p = measure.parity(tr, tn)
            rows.append({"tokenizer": name, "lang": lang, "flores_code": flores, "pairs": len(native),
                         "native_tokens": int(tn.sum()), "roman_tokens": int(tr.sum()),
                         "roman_vs_native": round(float(p["parity"]), 4),
                         "ci_low": round(p["parity_ci_low"], 4), "ci_high": round(p["parity_ci_high"], 4),
                         "native_tokens_per_word": round(tn.sum() / sum(measure.n_words(t) for t in native), 4),
                         "roman_tokens_per_word": round(tr.sum() / sum(measure.n_words(t) for t in roman), 4)})
    measure.RESULTS.mkdir(exist_ok=True)
    with open(measure.RESULTS / "dakshina.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    summary = {"source": URL, "md5_b64": MD5_B64, "license": "CC BY-SA 4.0", "tokenizers": meta,
               "roman_vs_native": {t: {r["lang"]: r["roman_vs_native"] for r in rows if r["tokenizer"] == t}
                                   for t in toks}}
    (measure.RESULTS / "dakshina_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                                           encoding="utf-8")
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
