"""Does the tokenizer training text contain FLORES-200 sentences? (Papers 1 and 2 list this as unchecked.)

For each language, every FLORES sentence (dev and devtest) is cut into overlapping n-grams: 13 words for scripts
written with spaces, 30 characters for scripts without (Chinese, Japanese, Thai, Lao, Khmer, Myanmar, Tibetan).
A sentence counts as contaminated if at least half of its n-grams appear anywhere in that language's training
sample (sentences shorter than one n-gram must appear whole). Text is NFC-normalised, lower-cased, whitespace
collapsed. The 13-gram rule follows the GPT-3 paper's contamination analysis.

Usage: python -m tokenizer_v3.contamination --samples DIR (samples/*.txt, NUL-separated documents)
          --flores flores200_dataset.tar.gz --out OUT [--workers 4]
"""
import argparse
import json
import re
import unicodedata
from multiprocessing import Pool
from pathlib import Path

from .eval_world import load_flores

NO_SPACE = ("Hans", "Hant", "Jpan", "Thai", "Laoo", "Khmr", "Mymr", "Tibt")
W, C = 13, 30
FLORES = {}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).lower()).strip()


def grams(s: str, chars: bool) -> list[str]:
    if chars:
        return [s[i:i + C] for i in range(0, max(1, len(s) - C + 1))] if len(s) >= C else [s]
    w = s.split()
    return [" ".join(w[i:i + W]) for i in range(0, max(1, len(w) - W + 1))] if len(w) >= W else [s]


def _init(flores: dict) -> None:
    FLORES.update(flores)


def check(args: tuple[str, str]) -> dict:
    lang, path = args
    chars = lang.split("_")[-1] in NO_SPACE
    sents = FLORES[lang]
    per = [set(grams(norm(s), chars)) for s in sents]
    wanted = set().union(*per)
    short = [g for g in wanted if (len(g) < C if chars else len(g.split()) < W)]  # whole short sentences
    found = set()
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    n_chars = len(text)
    for doc in text.split("\x00"):
        d = norm(doc)
        if not d:
            continue
        if chars:  # every 30-character window of the document, looked up in a set: linear in text length
            for i in range(len(d) - C + 1):
                g = d[i:i + C]
                if g in wanted:
                    found.add(g)
        else:
            w = d.split()
            for i in range(len(w) - W + 1):
                g = " ".join(w[i:i + W])
                if g in wanted:
                    found.add(g)
        for g in short:
            if g in d:
                found.add(g)
    hits = [i for i, gs in enumerate(per) if len(gs & found) >= 0.5 * len(gs)]
    return {"lang": lang, "sentences": len(sents), "contaminated": len(hits), "sample_chars": n_chars,
            "examples": [sents[i][:160] for i in hits[:3]]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--flores", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev, devtest = load_flores(a.flores, "dev"), load_flores(a.flores, "devtest")
    FLORES.update({l: dev[l] + devtest[l] for l in devtest})
    files = {f.stem: f for f in a.samples.rglob("*.txt")}
    jobs = [(l, str(files[l])) for l in sorted(FLORES) if l in files]
    print(f"{len(jobs)} languages with a training sample; {len(FLORES) - len(jobs)} without", flush=True)
    with Pool(a.workers, initializer=_init, initargs=(FLORES,)) as pool:  # initializer: works with spawn too
        res = []
        for r in pool.imap_unordered(check, jobs):
            res.append(r)
            print(f"{r['lang']}: {r['contaminated']}/{r['sentences']}", flush=True)
    res.sort(key=lambda r: -r["contaminated"])
    tot = sum(r["contaminated"] for r in res)
    summary = {"languages_checked": len(res), "sentences_checked": sum(r["sentences"] for r in res),
               "contaminated_sentences": tot, "languages_with_any": sum(r["contaminated"] > 0 for r in res),
               "rule": f"{W}-word or {C}-character n-grams; >= 50% of a sentence's n-grams found", "by_language": res}
    (a.out / "contamination.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_language"}, indent=1))


if __name__ == "__main__":
    main()
