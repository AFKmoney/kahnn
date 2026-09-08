# Kahnn pretrain corpus (`corpus.txt`)

Mixed **English prose + Python/code** corpus for `train_universal.py` nano
(and later commodity) pretraining on CPU. Built 2026-09-08 for Philippe-Antoine
Robert's Kahnn lifelong-learning path.

> **Git policy:** commit this `DATA.md` (and optionally small rebuild helpers).
> Do **not** commit `corpus.txt` (~109 MiB) or bulk `raw/` / `code/` dumps
> unless explicitly requested. Document path + provenance; rebuild locally.

## Local path (box)

| Artifact | Path | Size / tokens |
|----------|------|----------------|
| Training file | `/workspace/kahnn/data/corpus.txt` | **~109 MiB** (114072042 bytes) · **~46M** GPT-2 tokens estimated |
| Provenance sources | `data/raw/`, `data/code/` | kept locally for rebuild |
| This doc | `data/DATA.md` | in git |

Token estimate ratio ≈ **0.406** GPT-2 tokens/char (sampled `tiktoken` encode).

## Contents

| Part | Sources | License / terms |
|------|---------|-----------------|
| English prose (Gutenberg) | Project Gutenberg plain text: Pride and Prejudice, Alice in Wonderland, Frankenstein, Sherlock Holmes, Moby Dick, Tom Sawyer, A Tale of Two Cities, Dorian Gray, The Yellow Wallpaper, Dracula, The Prince, Huck Finn, Great Expectations, Jane Eyre, The Call of the Wild, Peter Pan, Jekyll & Hyde, Metamorphosis | Public domain (US) |
| TinyStories-style | `roneneldan/TinyStories` validation split (`TinyStories-valid.txt` via Hugging Face) | CDLA-Sharing-1.0 (dataset card) |
| Wikipedia extracts | Short REST API summaries for CS/ML topics | CC BY-SA 4.0 (Wikipedia) |
| Code — Kahnn | All `*.py` / `*.sh` / `*.md` from this repo (excluding `data/` and `runs/`) | Same as AFKmoney/kahnn |
| Code — CPython stdlib | Excerpts from `python/cpython` 3.12: `functools`, `dataclasses`, `pathlib`, `typing`, `json`, `heapq`, `bisect` | PSF License |

## Layout

- `data/raw/` — downloaded prose sources (kept for provenance; gitignored bulk)
- `data/code/` — code excerpts (gitignored bulk)
- `data/corpus.txt` — concatenated training file (prose + interleaved code; **gitignored**)
- `data/DATA.md` — this file (**in repo**)

## Build notes / how to rebuild

1. Download Gutenberg plain texts + TinyStories-valid + short Wikipedia summaries into `data/raw/`.
2. Collect Kahnn sources + CPython 3.12 excerpts into `data/code/`.
3. Concatenate prose + lightly interleaved/repeated code into a **base mix**
   (~38 MB, ~15.4M GPT-2 tokens estimated).
4. Repeat the base mix **3×** into `data/corpus.txt` so one `StreamingCorpus`
   pass covers nano Chinchilla (~37M tokens); final file ~109–114 MB / ~46M tokens.
5. Only use legally downloadable open data (no paywalled / private scrapes).

Code is interleaved and lightly repeated so long novels do not drown the mix.

Exact streamed token counts appear in `train_universal.py` logs
(see `docs/CPU_RUN_LOG.md` for the 2026-09-08 nano run).

## Usage

```bash
python train_universal.py \
  --data ./data/corpus.txt \
  --output ./runs/nano_base \
  --config nano --device auto
```
