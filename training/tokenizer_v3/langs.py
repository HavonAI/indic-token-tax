"""Language plan for Havon tokenizer v3: which text to sample and how much per language.

Budgets are in characters. Indian languages get the largest share (Havon's focus); about fifty major
world languages get a medium share; every other FLORES-200 language gets a small share so no script
is starved of vocabulary. All sources are human-written public text.
"""

# FLORES-200 code -> FineWeb-2 config, where the two datasets name a language differently.
FLORES_TO_FW2 = {
    "zho_Hans": "cmn_Hani", "zho_Hant": "cmn_Hani", "yue_Hant": "yue_Hani", "pes_Arab": "fas_Arab",
    "prs_Arab": "fas_Arab", "est_Latn": "ekk_Latn", "tgl_Latn": "fil_Latn", "grn_Latn": "gug_Latn",
    "aka_Latn": "twi_Latn", "kon_Latn": "kng_Latn",
}

# 22 scheduled languages of India (+ Nepali, Sinhala neighbours), with Sangraha verified folders.
INDIC = {
    "asm_Beng": "asm", "ben_Beng": "ben", "brx_Deva": "brx", "doi_Deva": "doi", "guj_Gujr": "guj",
    "hin_Deva": "hin", "kan_Knda": "kan", "kas_Arab": "kas", "gom_Deva": "gom", "mai_Deva": "mai",
    "mal_Mlym": "mal", "mni_Beng": "mni", "mar_Deva": "mar", "npi_Deva": "nep", "ory_Orya": "ori",
    "pan_Guru": "pan", "san_Deva": "san", "sat_Olck": "sat", "snd_Arab": "snd", "tam_Taml": "tam",
    "tel_Telu": "tel", "urd_Arab": "urd",
}

MAJOR = [
    "arb_Arab", "cmn_Hani", "yue_Hani", "spa_Latn", "fra_Latn", "deu_Latn", "por_Latn", "rus_Cyrl",
    "jpn_Jpan", "kor_Hang", "ita_Latn", "nld_Latn", "pol_Latn", "tur_Latn", "vie_Latn", "ind_Latn",
    "tha_Thai", "fas_Arab", "ukr_Cyrl", "ces_Latn", "ron_Latn", "hun_Latn", "ell_Grek", "swe_Latn",
    "dan_Latn", "fin_Latn", "nob_Latn", "heb_Hebr", "zsm_Latn", "fil_Latn", "swh_Latn", "amh_Ethi",
    "hau_Latn", "yor_Latn", "ibo_Latn", "zul_Latn", "bul_Cyrl", "srp_Cyrl", "hrv_Latn", "slk_Latn",
    "cat_Latn", "ekk_Latn", "lit_Latn", "lvs_Latn", "slv_Latn", "uzn_Latn", "kaz_Cyrl", "azj_Latn",
    "khm_Khmr", "lao_Laoo", "mya_Mymr", "sin_Sinh", "kat_Geor", "hye_Armn", "pbt_Arab", "som_Latn",
]

# v3.1 budgets (v3 used 30M / 10M / 2M / English 120M, trained at fraction 0.6): rebalanced toward
# English and the major world languages, where v3 trailed GPT-4o, while keeping the total near v3's
# trained size so BPE fits in Kaggle's 32 GB.
INDIC_CHARS = 20_000_000
MAJOR_CHARS = 15_000_000
MINOR_CHARS = 1_500_000
SPECIAL_SOURCES = {  # name -> (char budget, description)
    "eng": (200_000_000, "FineWeb-Edu English"),  # large share: target is parity with OpenAI on English
    "hin_rom": (20_000_000, "Hinglish: L3Cube HingCorpus + FineWeb-2 hin_Latn"),
    "mar_rom": (20_000_000, "Romanized Marathi: L3Cube MeCorpus + FineWeb-2 mar_Latn"),
    "math": (30_000_000, "FineMath + OpenWebMath"),
    "code": (40_000_000, "Python, permissive licenses only"),
}


def fw2_config(flores_code: str) -> str:
    return FLORES_TO_FW2.get(flores_code, flores_code)


def build_plan(flores_codes: list[str], fw2_configs: set[str]) -> dict[str, dict]:
    """Return {sample_name: {"kind", "chars", ...}} covering every FLORES language that has data."""
    plan: dict[str, dict] = {name: {"kind": "special", "chars": chars} for name, (chars, _) in SPECIAL_SOURCES.items()}
    # Include every Indian language even if FLORES lacks it (Bodo, Dogri, Konkani are not in FLORES-200).
    for code in sorted(set(flores_codes) | set(INDIC)):
        if code == "eng_Latn":
            continue
        cfg = fw2_config(code)
        if code in INDIC:
            plan[code] = {"kind": "indic", "chars": INDIC_CHARS, "sangraha": INDIC[code],
                          "fw2": cfg if cfg in fw2_configs else None}
        elif cfg in fw2_configs and cfg not in plan:
            plan[cfg] = {"kind": "world", "chars": MAJOR_CHARS if cfg in MAJOR else MINOR_CHARS, "fw2": cfg}
    return plan
