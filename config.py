"""
config.py
---------
Central configuration file. All path constants are resolved
relative to the project root so the project is portable.
"""

import os

# Project root = directory containing this file
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Data paths ───────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(PROJECT_ROOT, "data", "output")
CSV_PATH    = os.path.join(DATA_DIR,     "student_data.csv")

# ── Database paths ───────────────────────────────────────────────────────────
DB_DIR      = os.path.join(PROJECT_ROOT, "database")
DB_PATH     = os.path.join(DB_DIR,       "student_data.db")
SCHEMA_PATH = os.path.join(DB_DIR,       "schema.sql")

# ── Model artefact paths ─────────────────────────────────────────────────────
MODELS_DIR      = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH      = os.path.join(MODELS_DIR,   "best_model.pkl")
METRICS_PATH    = os.path.join(MODELS_DIR,   "model_metrics.json")
CLF_REPORT_PATH = os.path.join(MODELS_DIR,   "classification_reports.json")

# ── Report / output paths ────────────────────────────────────────────────────
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")
