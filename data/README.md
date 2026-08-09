# Data

This project uses the **PaySim** synthetic mobile-money fraud dataset
([Kaggle: Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)).

## Manual download

Automatic Kaggle authentication and downloading are intentionally **not** implemented.

1. Download the PaySim CSV from Kaggle (or another authorized source).
2. Place the CSV file under:

   ```text
   data/raw/
   ```

3. Ensure there is **exactly one** `.csv` file in `data/raw/`. The loader will use that file. If zero or multiple CSVs are present, loading fails with a clear error.

## Git policy

The raw dataset is **intentionally excluded from Git** (see repository `.gitignore`). Only this README and a directory placeholder are tracked under `data/`.
