"""
np_to_gemini.py

Utility to convert .npy files into formats easier for LLM/ML analysis (JSONL or CSV).
Place this script under the `diffusion/` folder and run it from the repository root or from inside `diffusion/`.

Features:
- Read a .npy file that contains a numpy array, Python list, or Python dict (saved with np.save(..., allow_pickle=True)).
- Convert and write:
  * JSONL: one JSON object per top-level element (if list/array) or one object for top-level dict (or one per key when --dict-entries).
  * CSV: if the object is a 1D/2D array or a dict of 1D arrays of equal length, write as CSV table.
- Handles numpy dtypes by converting to Python native types (via .tolist()).

Usage examples:
  python diffusion/np_to_gemini.py experiment_results.npy --out jsonl --out-file results.jsonl
  python diffusion/np_to_gemini.py data.npy --out csv --out-file data.csv

This is intentionally conservative: no smoothing or changes are performed to numeric data; this script only serializes.
"""

import argparse
import json
import os
import sys
import numpy as np
import csv
from typing import Any


def to_python(obj: Any) -> Any:
    """Recursively convert numpy types/arrays to Python native types that json can serialize."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    # fallback: try to convert
    try:
        if hasattr(obj, 'tolist'):
            return obj.tolist()
    except Exception:
        pass
    return obj


def save_jsonl(data: Any, out_path: str, dict_entries: bool = False):
    """Write data to JSONL. If data is list/ndarray -> one line per element. If dict and dict_entries True -> one line per key with {'key':..., 'value':...}."""
    with open(out_path, 'w', encoding='utf-8') as f:
        if isinstance(data, (list, tuple, np.ndarray)):
            for item in data:
                json.dump(to_python(item), f, ensure_ascii=False)
                f.write('\n')
            return
        if isinstance(data, dict):
            if dict_entries:
                for k, v in data.items():
                    json.dump({'key': k, 'value': to_python(v)}, f, ensure_ascii=False)
                    f.write('\n')
            else:
                json.dump(to_python(data), f, ensure_ascii=False)
                f.write('\n')
            return
        # fallback: single object -> single line
        json.dump(to_python(data), f, ensure_ascii=False)
        f.write('\n')


def save_csv(data: Any, out_path: str):
    """Write data to CSV. Supports:
    - 1D array -> one column
    - 2D array -> table
    - dict of 1D arrays (same length) -> columns by key
    """
    # If numpy array
    if isinstance(data, np.ndarray):
        arr = data
        if arr.ndim == 1:
            # one column
            with open(out_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['value'])
                for v in arr.tolist():
                    writer.writerow([v])
            return
        elif arr.ndim == 2:
            with open(out_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for row in arr.tolist():
                    writer.writerow(row)
            return
        else:
            raise ValueError('Can only save 1D or 2D numpy arrays to CSV.')

    # If list of scalars or lists
    if isinstance(data, list):
        # if list of lists / tuples -> rows
        if len(data) > 0 and isinstance(data[0], (list, tuple, np.ndarray)):
            with open(out_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for row in data:
                    writer.writerow([v for v in row])
            return
        # else treat as single column
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['value'])
            for v in data:
                writer.writerow([v])
        return

    # If dict of 1D arrays
    if isinstance(data, dict):
        keys = list(data.keys())
        col_lengths = [len(data[k]) if hasattr(data[k], '__len__') else None for k in keys]
        if all(isinstance(l, int) for l in col_lengths) and len(set(col_lengths)) == 1:
            n = col_lengths[0]
            with open(out_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(keys)
                for i in range(n):
                    row = [to_python(data[k])[i] for k in keys]
                    writer.writerow(row)
            return
        raise ValueError('Dict must contain 1D arrays of equal length to be saved as CSV.')

    raise ValueError('Unsupported data type for CSV output.')


def main():
    p = argparse.ArgumentParser(description='Convert .npy to JSONL/CSV for LLM analysis (Gemini).')
    p.add_argument('npy_file', help='.npy input file (may contain arrays, list, dict).')
    p.add_argument('--out', choices=['jsonl', 'csv'], default='jsonl', help='Output format')
    p.add_argument('--out-file', default=None, help='Output filename (defaults to input basename + extension)')
    p.add_argument('--dict-entries', action='store_true', help='When writing JSONL from dict, write one line per key (key,value)')
    args = p.parse_args()

    if not os.path.exists(args.npy_file):
        print('Input file not found:', args.npy_file, file=sys.stderr)
        sys.exit(2)

    data = np.load(args.npy_file, allow_pickle=True)

    # If saved as an array containing Python objects (e.g., list or dict), unwrap
    try:
        # if data is a 0-d numpy array with a Python object inside
        if isinstance(data, np.ndarray) and data.shape == ():
            data = data.item()
    except Exception:
        pass

    if args.out_file:
        out_file = args.out_file
    else:
        base = os.path.splitext(os.path.basename(args.npy_file))[0]
        out_file = base + ('.jsonl' if args.out == 'jsonl' else '.csv')

    print(f'Converting "{args.npy_file}" -> "{out_file}" as {args.out}')

    if args.out == 'jsonl':
        save_jsonl(data, out_file, dict_entries=args.dict_entries)
    else:
        save_csv(data, out_file)

    print('Done.')


if __name__ == '__main__':
    main()
