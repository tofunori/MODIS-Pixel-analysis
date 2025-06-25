import duckdb
import pandas as pd

# Connect to database
conn = duckdb.connect('/home/tofunori/duckdb-data/modis_analysis.db')

# Read the new CSV file
csv_path = "/home/tofunori/Projects/MODIS Pixel analysis/data/raw/MOD10A1_albedo_pixel_id_2010_2024 - MOD10A1_albedo_pixel_id_2010_2024.csv"
print("Reading CSV file...")
df = pd.read_csv(csv_path)

print(f"CSV loaded with {len(df)} rows and {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")

# Create new table from CSV
table_name = "modis_pixel_data"
print(f"Creating table '{table_name}'...")

# Register DataFrame with DuckDB and create table
conn.register('temp_df', df)
conn.execute(f"DROP TABLE IF EXISTS {table_name}")
conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_df")

print(f"Table '{table_name}' created successfully!")

# Show table info
print(f"\nTable schema:")
schema = conn.execute(f'DESCRIBE {table_name}').fetchall()
for col in schema:
    print(f"  {col[0]}: {col[1]}")

print(f"\nRow count: {conn.execute(f'SELECT COUNT(*) FROM {table_name}').fetchone()[0]}")

# Show sample data
print(f"\nSample data:")
sample = conn.execute(f'SELECT * FROM {table_name} LIMIT 3').fetchall()
for row in sample:
    print(f"  {row}")

conn.close()
print("Database connection closed.")