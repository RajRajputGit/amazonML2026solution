import sys

for mod in ['pandas', 'numpy', 'polars', 'pyarrow', 'duckdb', 'tqdm']:
    try:
        m = __import__(mod)
        print(f"{mod}: {getattr(m, '__version__', 'available')}")
    except ImportError:
        print(f"{mod}: NOT installed")
