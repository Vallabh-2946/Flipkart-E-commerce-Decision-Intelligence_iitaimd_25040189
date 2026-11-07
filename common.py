"""Shared CSV and formatting helpers for the Flipkart capstone."""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "records.csv"
OUTPUT_DIR = ROOT / "outputs"


def load_data():
    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames
        rows = list(reader)
    values = np.array([[float(row[name]) if name != "record_id" else row[name]
                        for name in names] for row in rows], dtype=object)
    numeric = values[:, 1:].astype(float)
    return names, rows, numeric


def write_results(path, items):
    OUTPUT_DIR.mkdir(exist_ok=True)
    with (OUTPUT_DIR / path).open("w", encoding="utf-8", newline="") as handle:
        for key, value in items:
            if isinstance(value, str):
                shown = value
            elif isinstance(value, (int, np.integer)):
                shown = str(int(value))
            else:
                shown = f"{float(value):.4f}"
            handle.write(f"{key} = {shown}\n")


def write_csv(path, header, rows):
    OUTPUT_DIR.mkdir(exist_ok=True)
    with (OUTPUT_DIR / path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
