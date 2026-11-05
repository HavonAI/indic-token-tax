"""Check whether DeepSeek's served V4 models share the DeepSeek-V3 tokenizer, with file hashes as evidence.

Compares, component by component, the tokenizer.json of DeepSeek-V3 (pinned revision used in the study)
against the current DeepSeek-V4 releases, and also checks that all of them encode a FLORES sample to
identical token ids. Writes results/tokenizer_equivalence.json.

Usage: python check_tokenizer_equivalence.py
"""
import hashlib
import json
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
REFERENCE = ("deepseek-ai/DeepSeek-V3", "e815299")
CANDIDATES = ["deepseek-ai/DeepSeek-V4.1-Flash", "deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Pro"]
COMPONENTS = ["normalizer", "pre_tokenizer", "decoder", "post_processor", "added_tokens"]
SAMPLE_LANGS = ["eng_Latn", "hin_Deva", "mar_Deva", "tam_Taml", "ory_Orya", "sat_Olck", "urd_Arab"]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj) -> str:
    return sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def describe(repo: str, revision: str | None) -> dict:
    resolved = HfApi().model_info(repo, revision=revision).sha
    path = hf_hub_download(repo, "tokenizer.json", revision=resolved)
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    parts = {c: canonical(data.get(c)) for c in COMPONENTS}
    parts["model.vocab"] = canonical(data["model"]["vocab"])
    parts["model.merges"] = canonical(data["model"]["merges"])
    return {"repo": repo, "revision": resolved, "file_sha256": sha256(raw), "components": parts, "path": path}


def flores_sample(n: int = 200) -> list[str]:
    tar_path = ROOT / "data" / "flores" / "flores200_dataset.tar.gz"
    texts = []
    with tarfile.open(tar_path) as tar:
        for lang in SAMPLE_LANGS:
            member = tar.getmember(f"./flores200_dataset/dev/{lang}.dev")
            texts += tar.extractfile(member).read().decode("utf-8").splitlines()[:n]
    return texts


def main() -> None:
    ref = describe(*REFERENCE)
    texts = flores_sample()
    ref_ids = [e.ids for e in Tokenizer.from_file(ref["path"]).encode_batch(texts, add_special_tokens=False)]
    report = {"reference": {k: v for k, v in ref.items() if k != "path"}, "sample_sentences": len(texts),
              "sample_languages": SAMPLE_LANGS, "candidates": []}
    for repo in CANDIDATES:
        try:
            cand = describe(repo, None)
        except Exception as exc:  # a repo may be gated or have no tokenizer.json; record that instead
            report["candidates"].append({"repo": repo, "error": f"{type(exc).__name__}: {exc}"[:300]})
            continue
        ids = [e.ids for e in Tokenizer.from_file(cand["path"]).encode_batch(texts, add_special_tokens=False)]
        same = {c: cand["components"][c] == ref["components"][c] for c in ref["components"]}
        report["candidates"].append({
            **{k: v for k, v in cand.items() if k != "path"},
            "identical_file": cand["file_sha256"] == ref["file_sha256"],
            "identical_components": same,
            "identical_encodings": sum(a == b for a, b in zip(ids, ref_ids)),
        })
    out = ROOT / "results" / "tokenizer_equivalence.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for c in report["candidates"]:
        if "error" in c:
            print(c["repo"], "ERROR", c["error"][:120])
        else:
            print(c["repo"], c["revision"][:7], "file identical:", c["identical_file"], "| differing components:",
                  [k for k, v in c["identical_components"].items() if not v] or "none",
                  f"| identical encodings {c['identical_encodings']}/{len(texts)}")


if __name__ == "__main__":
    main()
