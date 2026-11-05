"""Text sources per language. Every source is human-written public text; nothing is model-generated.

Language keys:
  mar      Marathi, Devanagari script
  hin      Hindi, Devanagari script
  eng      English
  hin_rom  Hinglish: Hindi typed in Roman letters, often mixed with English
  mar_rom  Marathi typed in Roman letters, often mixed with English
  math     Web pages with mathematics (worked solutions, proofs, notation)
  code     Python source files under permissive licenses
"""
import glob
import html
import itertools
import os
import re
import tarfile
from collections.abc import Callable, Iterator

MIN_CHARS = 200  # drop very short web pages (menus, fragments)
TWEETS_PER_DOC = 32  # social posts are short, so group them into document-sized chunks
RAW_DIR = os.environ.get("HAVON_RAW_DIR", "/kaggle/input")
HINGCORPUS = ("hinglish-orthobench/HingCorpus", "L3Cube-HingCorpus_roman.tar.gz")
MECORPUS_GLOB = "**/me_roman*"
PYTHON_CODE = "hf://datasets/codeparrot/github-code-clean@refs%2Fconvert%2Fparquet/Python-all/partial-train/*.parquet"
PERMISSIVE = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "cc0-1.0", "unlicense"}
MAX_CODE_CHARS = 100_000  # skip huge generated or data files
# Basic safety filter for Roman-script web and social text, which carries more adult and abusive content.
BLOCKLIST = re.compile(r"\b(chudai|sex ?story|xxx|porn|lund|chut|randi|bhosd\w*|madarchod|behen ?chod|bc mc)\b",
                       re.IGNORECASE)
_MENTION = re.compile(r"@\w+")
_URL = re.compile(r"https?://\S+|www\.\S+")


def clean_post(line: str) -> str | None:
    text = html.unescape(line.strip().strip('"'))
    text = _URL.sub("", _MENTION.sub("", text))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 10 or text.lower() == "text" or BLOCKLIST.search(text):
        return None
    return text


def group_posts(lines: Iterator[str]) -> Iterator[str]:
    cleaned = (t for t in map(clean_post, lines) if t)
    while chunk := list(itertools.islice(cleaned, TWEETS_PER_DOC)):
        yield "\n".join(chunk)


def hf_stream(repo: str, safe_filter: bool = False, **kw) -> Callable[[], Iterator[str]]:
    def gen() -> Iterator[str]:
        from datasets import load_dataset
        for row in load_dataset(repo, split="train", streaming=True, **kw):
            text = row.get("text")
            if not text or len(text) < MIN_CHARS or (safe_filter and BLOCKLIST.search(text)):
                continue
            yield text
    return gen


def python_code() -> Iterator[str]:
    from datasets import load_dataset
    for row in load_dataset("parquet", data_files=PYTHON_CODE, split="train", streaming=True):
        code = row.get("code") or ""
        if row.get("license") in PERMISSIVE and MIN_CHARS <= len(code) <= MAX_CODE_CHARS:
            yield code


def hingcorpus() -> Iterator[str]:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=HINGCORPUS[0], filename=HINGCORPUS[1], repo_type="dataset")
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".txt"):
                lines = (b.decode("utf-8", "ignore") for b in tar.extractfile(member))
                yield from group_posts(lines)


def mecorpus() -> Iterator[str]:
    hits = sorted(glob.glob(os.path.join(RAW_DIR, MECORPUS_GLOB), recursive=True))
    if not hits:
        raise FileNotFoundError(f"MeCorpus file (me_roman) not found under {RAW_DIR}")
    with open(hits[0], encoding="utf-8", errors="ignore") as f:
        yield from group_posts(f)


SOURCES: dict[str, list[Callable[[], Iterator[str]]]] = {
    "mar": [hf_stream("ai4bharat/sangraha", data_dir="verified/mar"),
            hf_stream("HuggingFaceFW/fineweb-2", name="mar_Deva")],
    "hin": [hf_stream("ai4bharat/sangraha", data_dir="verified/hin"),
            hf_stream("HuggingFaceFW/fineweb-2", name="hin_Deva")],
    "eng": [hf_stream("HuggingFaceFW/fineweb-edu", name="sample-10BT")],
    "hin_rom": [hingcorpus, hf_stream("HuggingFaceFW/fineweb-2", safe_filter=True, name="hin_Latn")],
    "mar_rom": [mecorpus, hf_stream("HuggingFaceFW/fineweb-2", safe_filter=True, name="mar_Latn")],
    "math": [hf_stream("HuggingFaceTB/finemath", name="finemath-3plus"), hf_stream("open-web-math/open-web-math")],
    "code": [python_code],
}


def stream(lang: str) -> Iterator[str]:
    """Yield documents, alternating between the language's sources until all are exhausted."""
    for docs in itertools.zip_longest(*(src() for src in SOURCES[lang])):
        yield from (d for d in docs if d is not None)
