"""Rupee cost of the token tax: a worked example from measured parity ratios and official API prices.

Inputs: results/results.csv and results/indic_parity.csv (from measure.py). Prices are copied by hand
from the official pricing pages listed in PRICES (accessed 2026-11-05) and are list prices for the
standard tier, with no caching, batching or discounts. USD prices are converted with the FBIL
reference rate. Nothing here calls a model API.

Usage: python cost.py
Outputs: results/cost.csv, results/cost_summary.json
"""
from __future__ import annotations

import csv
import json

from measure import DATA, FLORES_INDIC, HF_TOKENIZERS, RESULTS

ACCESSED = "2026-11-05"
USD_INR = 94.3996  # FBIL reference rate, INR per 1 USD, as at 1:00 pm IST on 5 Nov 2026
FX_SOURCE = "https://www.rbi.org.in/ (Exchange Rates panel, 'Source: FBIL', as at 1.00pm of November 05, 2026)"

OPENAI = "https://developers.openai.com/api/docs/pricing"
DEEPSEEK = "https://api-docs.deepseek.com/quick_start/pricing"
SARVAM = "https://www.sarvam.ai/api-pricing"
# model -> (tokenizer measured, currency, input per 1M tokens, output per 1M tokens, source, note)
PRICES = {
    "GPT-4o": ("GPT-4o (o200k_base)", "USD", 2.50, 10.00, OPENAI, "standard tier"),
    "GPT-4.1 mini": ("GPT-4o (o200k_base)", "USD", 0.40, 1.60, OPENAI, "standard tier"),
    "GPT-5": ("GPT-4o (o200k_base)", "USD", 1.25, 10.00, OPENAI, "standard tier"),
    "GPT-5 mini": ("GPT-4o (o200k_base)", "USD", 0.25, 2.00, OPENAI, "standard tier"),
    "DeepSeek deepseek-flash": ("DeepSeek-V3", "USD", 0.30, 1.20, DEEPSEEK,
                                "DeepSeek-V4.1-Flash-1017, peak-hour cache-miss rate (off-peak is half)"),
    "DeepSeek deepseek-v4-pro": ("DeepSeek-V3", "USD", 1.32, 3.96, DEEPSEEK,
                                 "DeepSeek-V4-Pro-0813, peak-hour cache-miss rate (off-peak is half)"),
    "Sarvam 105B (Sarvam API)": ("Sarvam-105B", "INR", 15.00, 60.00, SARVAM, "per 1M tokens, INR"),
}
# Measured tokenizers used for model families that are newer than the tokenizer repo we pin.
# tiktoken 0.12.0 maps gpt-4o, gpt-4.1(-mini), gpt-5 and gpt-5-mini to o200k_base.
# DeepSeek V4 tokenizers below share DeepSeek-V3's vocabulary and merges (checked by verify_deepseek).
DEEPSEEK_V4 = {"deepseek-ai/DeepSeek-V4.1-Flash-1017": "1bae0e1088993b9b82de08f59de5be5a34e42170",
               "deepseek-ai/DeepSeek-V4-Pro-0813": "72e1d3230f6c080a530b0a1d46f8eb4602340597"}

# Worked example (assumptions, not measurements)
CHATS_PER_MONTH = 100_000
EN_INPUT_TOKENS = 800   # per chat, English-equivalent, all user-visible conversation text
EN_OUTPUT_TOKENS = 400  # per chat, English-equivalent
EXAMPLE_LANGS = ["eng", "hin_Deva", "mar_Deva", "ben_Beng", "tam_Taml", "tel_Telu", "guj_Gujr",
                 "kan_Knda", "mal_Mlym", "ory_Orya", "pan_Guru", "urd_Arab", "asm_Beng", "hin_rom"]


def verify_deepseek() -> dict:
    from huggingface_hub import hf_hub_download

    def model(repo: str, rev: str) -> dict:
        p = hf_hub_download(repo, "tokenizer.json", revision=rev, cache_dir=str(DATA / "hf"))
        with open(p, encoding="utf-8") as f:
            j = json.load(f)
        return {k: j[k] for k in ("normalizer", "pre_tokenizer")} | {
            "vocab": j["model"]["vocab"], "merges": j["model"]["merges"]}

    ref = model(*HF_TOKENIZERS["DeepSeek-V3"])
    out = {}
    for repo, rev in DEEPSEEK_V4.items():
        same = model(repo, rev) == ref
        if not same:
            raise ValueError(f"{repo} tokenizer differs from DeepSeek-V3; parity cannot be reused")
        out[f"{repo}@{rev}"] = "vocab, merges, normalizer and pre-tokenizer identical to DeepSeek-V3"
    return out


def parities() -> dict:
    """tokenizer -> lang -> parity (own English baseline). hin_rom = Hinglish on PHINC."""
    par: dict[str, dict[str, float]] = {}
    with open(RESULTS / "indic_parity.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            par.setdefault(r["tokenizer"], {"eng": 1.0})[r["lang"]] = float(r["parity"])
    with open(RESULTS / "results.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == "phinc" and r["lang"] == "hin_rom":
                par[r["tokenizer"]]["hin_rom"] = float(r["parity"])
    return par


def main() -> None:
    ds = verify_deepseek()
    par = parities()
    names = {"eng": "English", "hin_rom": "Hinglish (PHINC)", **FLORES_INDIC}
    rows = []
    for model, (tok, cur, pin, pout, src, note) in PRICES.items():
        fx = USD_INR if cur == "USD" else 1.0
        en_inr = CHATS_PER_MONTH * (EN_INPUT_TOKENS * pin + EN_OUTPUT_TOKENS * pout) / 1e6 * fx
        for lang in list(FLORES_INDIC) + ["eng", "hin_rom"]:
            p = par[tok][lang]
            rows.append({"model": model, "tokenizer": tok, "lang": lang, "language": names[lang],
                         "parity": p, "price_in_per_1M": pin, "price_out_per_1M": pout, "currency": cur,
                         "inr_per_month": round(en_inr * p, 2),
                         "extra_inr_vs_english": round(en_inr * (p - 1), 2)})
    with open(RESULTS / "cost.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    table = {m: {r["lang"]: r["inr_per_month"] for r in rows if r["model"] == m and r["lang"] in EXAMPLE_LANGS}
             for m in PRICES}
    summary = {"accessed": ACCESSED, "usd_inr": USD_INR, "fx_source": FX_SOURCE,
               "assumptions": {"chats_per_month": CHATS_PER_MONTH,
                               "english_input_tokens_per_chat": EN_INPUT_TOKENS,
                               "english_output_tokens_per_chat": EN_OUTPUT_TOKENS,
                               "rule": "cost(L) = cost(English) x parity(L), applied to input and output",
                               "excludes": "system prompts, chat templates, caching, batch and off-peak discounts, taxes"},
               "prices": {m: {"tokenizer": t, "currency": c, "input_per_1M": i, "output_per_1M": o,
                              "source": s, "note": n} for m, (t, c, i, o, s, n) in PRICES.items()},
               "tokenizer_checks": {"openai": "tiktoken 0.12.0 encoding_name_for_model -> o200k_base for "
                                              "gpt-4o, gpt-4.1-mini, gpt-5, gpt-5-mini",
                                    "deepseek": ds,
                                    "sarvam": "assumed: Sarvam API 'Sarvam 105B' uses the tokenizer of "
                                              "sarvamai/sarvam-105b at the pinned revision (not verified)"},
               "inr_per_month": table}
    (RESULTS / "cost_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    for m, t in table.items():
        print(m, {k: round(v) for k, v in t.items()})


if __name__ == "__main__":
    main()
