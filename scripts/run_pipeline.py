"""CLI: python scripts/run_pipeline.py [--fast] [--root PATH] [--customers N] [--no-mlflow]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import run_pipeline  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the full RetailIQ pipeline")
    ap.add_argument("--fast", action="store_true", help="small hyper-parameter grids")
    ap.add_argument("--root", default=None, help="project root (default: repo root)")
    ap.add_argument("--customers", type=int, default=2000, help="synthetic customers if no real data is present")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    print(run_pipeline(a.root, fast=a.fast, n_customers=a.customers, track=not a.no_mlflow))
