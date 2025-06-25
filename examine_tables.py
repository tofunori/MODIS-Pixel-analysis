import duckdb

conn = duckdb.connect('/home/tofunori/duckdb-data/modis_analysis.db')

print("=== MODIS_DATA TABLE ===")
print("Schema:")
schema1 = conn.execute('DESCRIBE modis_data').fetchall()
for col in schema1:
    print(f"  {col[0]}: {col[1]}")

print(f"\nRow count: {conn.execute('SELECT COUNT(*) FROM modis_data').fetchone()[0]}")
print("\nSample data:")
sample1 = conn.execute('SELECT * FROM modis_data LIMIT 3').fetchall()
for row in sample1:
    print(f"  {row}")

print("\n" + "="*50)
print("=== MODIS_PIXEL_DATA TABLE ===")
print("Schema:")
schema2 = conn.execute('DESCRIBE modis_pixel_data').fetchall()
for col in schema2:
    print(f"  {col[0]}: {col[1]}")

print(f"\nRow count: {conn.execute('SELECT COUNT(*) FROM modis_pixel_data').fetchone()[0]}")
print("\nSample data:")
sample2 = conn.execute('SELECT * FROM modis_pixel_data LIMIT 3').fetchall()
for row in sample2:
    print(f"  {row}")

print("\n" + "="*50)
print("=== COMMON COLUMNS ===")
cols1 = set([col[0] for col in schema1])
cols2 = set([col[0] for col in schema2])
common = cols1.intersection(cols2)
only_in_data = cols1 - cols2
only_in_pixel = cols2 - cols1

print(f"Common columns ({len(common)}):")
for col in sorted(common):
    print(f"  - {col}")

print(f"\nOnly in modis_data ({len(only_in_data)}):")
for col in sorted(only_in_data):
    print(f"  - {col}")

print(f"\nOnly in modis_pixel_data ({len(only_in_pixel)}):")
for col in sorted(only_in_pixel):
    print(f"  - {col}")

conn.close()