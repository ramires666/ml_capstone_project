
import pandas as pd
from pathlib import Path

p = Path("data/processed/fundings.parquet")
if p.exists():
    df = pd.read_parquet(p)
    print(f"Columns in {p}: {df.columns.tolist()}")
else:
    print(f"File not found: {p}")
