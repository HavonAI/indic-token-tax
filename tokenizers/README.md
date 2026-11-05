# Havon tokenizers used in the study

Both are byte-level BPE tokenizers trained from scratch. The files are unchanged from training.

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
