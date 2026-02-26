import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np
import gc

print("Creating EDA sample...")
# Read the parquet file in row groups or just a few chunks to create a representative sample
pf = pq.ParquetFile('processed/all_data_full.parquet')
sample_tables = []

# There are multiple row groups, we will sample the first 5 row groups (or read 5% from the whole)
# Since we need it to be memory efficient, we can just read first 2.5 million rows and shuffle them
# The original data might be sorted by files, so reading scattered row groups is better

indices = np.linspace(0, pf.num_row_groups - 1, 10, dtype=int)
for i in indices:
    print(f"Reading row group {i} for sample...")
    rg = pf.read_row_group(i)
    # Take 20% of this row group
    df = rg.to_pandas()
    sample_df = df.sample(frac=0.2, random_state=42)
    sample_tables.append(pa.Table.from_pandas(sample_df))
    del df, rg, sample_df
    gc.collect()

sample_table = pa.concat_tables(sample_tables)
print(f"Saving {sample_table.num_rows} rows to sample...")
pq.write_table(sample_table, 'processed/eda_sample.parquet', compression='snappy')
print("Sample created successfully.")
