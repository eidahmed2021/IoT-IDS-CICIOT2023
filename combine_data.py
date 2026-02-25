import pandas as pd
import glob
import os
import pyarrow as pa
import pyarrow.parquet as pq
import gc

# 1. Get all files
files = glob.glob(r'd:\Ml Project\data\*.csv')
print(f'Found {len(files)} files to process.')

# 2. Output setup
os.makedirs('processed', exist_ok=True)
output_path = 'processed/all_data_full.parquet'
if os.path.exists(output_path):
    os.remove(output_path)

first_chunk = True
total_rows = 0
chunknumber = 0
writer = None

# We read in chunks of 50,000 rows
chunksize = 50000

for file_idx, f in enumerate(files):
    print(f"[{file_idx+1}/{len(files)}] Processing {os.path.basename(f)}...", end=" ")
    
    # Read file in chunks
    for chunk in pd.read_csv(f, chunksize=chunksize):
        # Data is kept as original types (no float32 downcasting)

        # Convert to PyArrow Table
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        
        # Initialize writer on the very first chunk
        if first_chunk:
            # Create a Parquet writer with snappy compression
            writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
            first_chunk = False
        
        # Write the chunk to the parquet file
        writer.write_table(table)
        
        total_rows += len(chunk)
        chunknumber += 1
        
        # Force garbage collection occasionally
        if chunknumber % 10 == 0:
            gc.collect()
            
    print(f"(Total rows so far: {total_rows:,})")

# Close the writer
if writer:
    writer.close()

print(f'\nSuccess! Successfully processed and combined {total_rows:,} rows into {output_path}')
print('Data types were kept as original.')
