# Havon tokenizers used in the study

**License:** CC BY-NC-SA 4.0 (see `LICENSE` in this folder). The training text includes L3Cube HingCorpus and MeCorpus, which are CC BY-NC-SA 4.0.

All three are byte-level BPE tokenizers trained from scratch. The files are unchanged from training.

| | Havon v1 | Havon v2 |
|---|---|---|
| File | `havon_v1/tokenizer.json` | `havon_v2/tokenizer.json` |
| SHA-256 | `d1cf905f328c9c2f9bc1f79bdd303111bb5df3705ef28719c0f430ef125b7474` | `098d13988eb2203bbaf38186180e29a87e804e635179cef4910610eb59f51d47` |
| Vocabulary | 32,768 | 32,768 |
| Normalizer | NFC | NFC |
| Pre-tokenizer | regex split, then byte-level (no regex) | same, but numbers split into single digits |
| Split regex | ``[^\r\n\p{L}\p{M}\p{N}]?[\p{L}\p{M}]+`` \| ``\p{N}{1,3}`` \| `` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*`` \| ``\s*[\r\n]+`` \| ``\s+(?!\S)`` \| ``\s+`` | same with ``\p{N}`` in place of ``\p{N}{1,3}`` |
| min_frequency | 2 | 2 |
| Special tokens | 5 | 32 (chat, reasoning and tool-call markers, plus 20 reserved) |
| Training sample | 200M characters from each of 5 domains | 150M characters from each of 7 domains |
| Domains | Marathi, Hindi, English, romanized Hindi, romanized Marathi | v1's five plus mathematics and Python code |

The regex keeps letters and combining marks (Unicode category M) together. This stops words being split at Devanagari vowel signs, which GPT-2's rule does.

## Training sources (human-written text only)

- **Marathi and Hindi:** AI4Bharat Sangraha (`verified/mar`, `verified/hin`), alternated with FineWeb-2 (`mar_Deva`, `hin_Deva`)
- **English:** FineWeb-Edu (`sample-10BT`)
- **Romanized Hindi:** L3Cube HingCorpus, alternated with FineWeb-2 `hin_Latn` (with a basic adult and abuse word filter)
- **Romanized Marathi:** L3Cube MeCorpus, alternated with FineWeb-2 `mar_Latn` (same filter)
- **Mathematics (v2 only):** FineMath (`finemath-3plus`) and OpenWebMath
- **Code (v2 only):** codeparrot/github-code-clean, Python only, restricted to MIT, Apache-2.0, BSD, ISC, CC0 and Unlicense files

Documents shorter than 200 characters are dropped. Social media posts are cleaned (HTML entities unescaped, mentions and URLs removed) and grouped 32 per document. Each domain's sample is the first documents streamed from its sources, up to that domain's character budget.

## Overlap with the evaluation data

HingCorpus and MeCorpus were used in training, so the romanized results on those corpora are optimistic. PHINC and Dakshina were not used in training. We did not check FLORES-200 sentences for overlap with the web-crawled training sources.

## Havon v3 (multilingual)

| | Havon v3 |
|---|---|
| File | `havon_v3/tokenizer.json` |
| SHA-256 | `b6dcb2360980386bee48ff733ed92cd3ed19a86d11df63331243efb9b237a721` |
| Vocabulary | 262,144 (64 special tokens: v2's 12 plus 52 reserved) |
| Normalizer | NFC |
| Pre-tokenizer | v2's rule (letters and marks together, single digits), plus chunking for scripts written without spaces: Han and kana in runs of at most 3 characters; Thai, Lao, Khmer, Myanmar and Tibetan in runs of at most 6 letters, each with its marks |
| min_frequency / max_token_length | 2 / 32 |
| Training sample | 1.69 billion characters planned; the first 60% of each language's sample was used (about 1.0 billion characters) to fit in 32 GB |
| Sources | FineWeb-2 (about 195 languages), AI4Bharat Sangraha plus FineWeb-2 (22 languages of India), FineWeb-Edu (English), and v2's romanized, mathematics and code sources. Human-written text only |
| Per-language budgets | 30M characters per language of India, 10M for about 56 major world languages, 2M for the rest, 120M English, 20M each for romanized Hindi and romanized Marathi, 30M mathematics, 40M code (before the 60% cut) |

FineWeb-2 and Sangraha are web text, and we did not check FLORES-200 sentences for overlap with v3's sample.

## Havon v4 (multi-word)

v4 keeps the first merges of Havon v3 and continues training with a pre-tokenizer that no longer splits at spaces, so later tokens can span several words (SuperBPE, Liu et al., 2025). Digits and punctuation runs are still split, and scripts without spaces keep v3's chunking. Stage 2 learned from the first 12% of each language's v3 sample (about 335 MB), using the SuperBPE fork of the Hugging Face tokenizers library at commit 757f2a5. Each smaller size is an exact prefix of a larger run's merges, cut with `truncate.py`.

| | v4-200k | v4-256k | v4-512k |
|---|---|---|---|
| File | `havon_v4/tokenizer_200k.json` | `havon_v4/tokenizer_256k.json` | `havon_v4/tokenizer_512k.json` |
| SHA-256 | `3d4f742314b67ceb224ddfe306611bf795ba8f407b726def7dc2cd0e78bf6eaf` | `424929c501cfdecdd4899aceb6d653a065ebd1fe7c6643d53ca77976a19a1758` | `7264c95c4b061e9f2babed1b8d3ae447ef674c770f3ef571546cab531a9f4bcd` |
| Vocabulary | 200,019 (equal to GPT-4o's o200k_base) | 262,144 | 524,288 |
| Merges inherited from v3 | 160,000 | 210,000 | 210,000 |
| Training run | its own run, trained to 200,019 | cut from the 524,288 run | the 524,288 run |

Special tokens, normalizer and the within-word rule are the same as v3's.
