"""
Configuration module for AdMIRe 2.0 Text-Only Bottleneck Architecture.

Loads secrets from a ``.env`` file via python-dotenv and exposes model
identifiers, API keys, and tunable hyper-parameters.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

CROSS_ENCODER_MODEL: str = "cross-encoder/stsb-distilroberta-base"
MSMARCO_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
LLM_TEMPERATURE: float = 0.0
def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default

LLM_MAX_TOKENS: int = _env_int("LLM_MAX_TOKENS", 600)
def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default

LLM_MIN_REQUEST_INTERVAL_SEC: float = _env_float("LLM_MIN_REQUEST_INTERVAL_SEC", 0.0)

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")

LLM_TOP3_RERANK_ENABLED: bool = _env_bool("LLM_TOP3_RERANK_ENABLED", True)
LLM_TOP3_PROB_SPREAD_MAX: float = _env_float("LLM_TOP3_PROB_SPREAD_MAX", 0.06)
LLM_TOP3_RERANK_MAX_TOKENS: int = _env_int("LLM_TOP3_RERANK_MAX_TOKENS", 350)

def _detect_device() -> str:
    override = os.getenv("DEVICE")
    if override:
        return override
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            t = torch.zeros(1, device="mps")
            return "mps"
    except Exception:
        pass
    return "cpu"

DEVICE: str = _detect_device()

DATA_DIR: str = str(_PROJECT_ROOT.parent / "data")
DEFAULT_INPUT_TSV: str = os.path.join(DATA_DIR, "submission_Chinese.tsv")
OUTPUT_DIR: str = str(_PROJECT_ROOT / "outputs")
DEFAULT_OUTPUT_TSV: str = os.path.join(OUTPUT_DIR, "predictions.tsv")
