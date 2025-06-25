import duckdb

conn = duckdb.connect('/home/tofunori/duckdb-data/modis_analysis.db')

print('Tables:')
tables = conn.execute('SHOW TABLES').fetchall()
print(tables)

if tables:
    table_name = tables[0][0]
    print(f'\nSchema for table {table_name}:')
    schema = conn.execute(f'DESCRIBE {table_name}').fetchall()
    for col in schema:
        print(f"  {col[0]}: {col[1]}")
    
    print(f'\nSample data from {table_name}:')
    sample = conn.execute(f'SELECT * FROM {table_name} LIMIT 5').fetchall()
    columns = [col[0] for col in schema]
    print(f"Columns: {columns}")
    for row in sample:
        print(f"  {row}")

conn.close()