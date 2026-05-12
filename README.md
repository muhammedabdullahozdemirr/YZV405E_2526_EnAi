# admire_text_bottleneck

A text-only baseline for the **AdMIRe 2.0** multilingual idiom-aware image ranking task.

Given a sentence in some language containing a potentially idiomatic compound, the system ranks 5 candidate image captions by how well they match the *intended* meaning of the sentence — literal or figurative.

## Pipeline

```
sentence  ──►  LLM unpacker  ──►  cross-encoder ranker  ──►  ranked captions
                                       (stsb + ms-marco)
```

1. **LLM unpacker** (`src/llm_unpacker.py`) — sends the sentence to OpenAI and returns a JSON object with:
   - `usage_type` (`figurative` | `literal`)
   - `english_translation`
   - `literal_meaning`
   - `context_visual` (real-world scene the sentence describes)
   - `compound_visual` (literal picture of the idiom itself)
   - `de_idiomatized_sentence` (figurative cases only)

2. **Ranker** (`src/ranker.py`) — scores each caption against the unpacker outputs using two cross-encoders and combines them with `usage_type`-aware weights:

   | signal | figurative | literal |
   |---|---|---|
   | `context_visual` | 0.55 | 0.20 |
   | `compound_visual` | 0.00 | 0.45 |
   | `literal_meaning` | 0.15 | 0.25 |
   | `english_translation` | 0.05 | 0.10 |
   | `de_idiomatized_sentence` | 0.25 | — |
   | stsb model weight | 0.80 | 0.30 |
   | ms-marco model weight | 0.20 | 0.70 |

   Includes a low-confidence fallback that re-scores using the dominant signal when the top two candidates are too close.

3. **Evaluator** (`src/evaluate.py`) — reports **NDCG@5 (3/1/0/0/0)** as the primary metric, plus top-1 accuracy and average Spearman ρ.

## Setup

```bash
git clone <repo-url>
cd admire_text_bottleneck
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
```

## Usage

```bash
# Run all language files end-to-end and produce submission files
python main.py

# Run a single language
python main.py --lang Turkish
python main.py --lang Chinese

# Quick benchmark: 20 Chinese rows + 30 Turkish rows
python main.py --mini-benchmarks
```

Outputs land in `outputs/`:

```
outputs/
├── predictions_Chinese.tsv
├── predictions_Turkish.tsv
├── ...
└── submission_XX.tsv     ← Codabench-ready
```

## Project layout

```
admire_text_bottleneck/
├── main.py
├── requirements.txt
├── benchmarks/
│   ├── ground_truth_chinese_20.json
│   └── ground_truth_turkish_30.json
├── data/                      # submission_*.tsv input files
├── outputs/
└── src/
    ├── config.py
    ├── data_loader.py
    ├── llm_unpacker.py
    ├── ranker.py
    ├── pipeline.py
    └── evaluate.py
```

## Models

| Role | Model |
|---|---|
| LLM | `gpt-4o-mini` |
| Cross-encoder (semantic similarity) | `cross-encoder/stsb-distilroberta-base` |
| Cross-encoder (passage ranking) | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

## Notes

- The LLM is non-deterministic even at `temperature=0`, so results can vary across runs by a few percentage points.
- Rate limiting is controlled by `LLM_MIN_REQUEST_INTERVAL_SEC` in `config.py` (default `0`).
- Cross-encoders run on Apple Silicon (`mps`) when available, otherwise on CPU.