# dans train.py
#
# Usage : python src/train.py

from pathlib import Path
from preprocess import preprocess
import yaml, pandas as pd
# from notebooks.jbtools import describoost

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "config.yaml"))
clean = Path(cfg["paths"]["clean"])
if not clean.exists():
    preprocess(cfg).to_csv(clean, index=False)
df = pd.read_csv(clean)
#describoost(df)
