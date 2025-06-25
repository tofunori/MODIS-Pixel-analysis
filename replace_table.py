import duckdb

# Connect to database
conn = duckdb.connect('/home/tofunori/duckdb-data/modis_analysis.db')

print("Replacing modis_data with enhanced modis_pixel_data...")

# Drop the old table and rename the new one
conn.execute("DROP TABLE modis_data")
conn.execute("ALTER TABLE modis_pixel_data RENAME TO modis_data")

print("Table replacement complete!")

# Verify the operation
print("\nVerifying new modis_data table:")
schema = conn.execute('DESCRIBE modis_data').fetchall()
print(f"Columns ({len(schema)}):")
for col in schema:
    print(f"  {col[0]}: {col[1]}")

row_count = conn.execute('SELECT COUNT(*) FROM modis_data').fetchone()[0]
print(f"\nRow count: {row_count}")

# Show available tables
print("\nAll tables in database:")
tables = conn.execute('SHOW TABLES').fetchall()
for table in tables:
    print(f"  - {table[0]}")

conn.close()
print("\nDatabase operation completed successfully!")