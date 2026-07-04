import sqlite3
import json
import os

def seed_domains():
    db_path = 'data/atlas.db'
    json_path = 'data/domain_examples.json'
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Connect to SQLite database (this will create it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    print("Creating tables...")
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS domains (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS domain_examples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        domain_id INTEGER NOT NULL,
        example_text TEXT NOT NULL,
        FOREIGN KEY(domain_id) REFERENCES domains(id)
    )
    ''')
    
    # Read the JSON file
    print(f"Reading {json_path}...")
    try:
        with open(json_path, 'r') as f:
            domain_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_path} not found.")
        return

    # Insert data
    print("Inserting data into database...")
    for domain_name, examples in domain_data.items():
        # Insert or ignore the domain to get its ID
        cursor.execute('''
        INSERT OR IGNORE INTO domains (name) VALUES (?)
        ''', (domain_name,))
        
        # Fetch the domain ID
        cursor.execute('SELECT id FROM domains WHERE name = ?', (domain_name,))
        domain_id = cursor.fetchone()[0]
        
        # Insert all examples for this domain
        for example_text in examples:
            cursor.execute('''
            INSERT INTO domain_examples (domain_id, example_text)
            SELECT ?, ? 
            WHERE NOT EXISTS (
                SELECT 1 FROM domain_examples 
                WHERE domain_id = ? AND example_text = ?
            )
            ''', (domain_id, example_text, domain_id, example_text))
            
    # Commit changes and close connection
    conn.commit()
    conn.close()
    print(f"Successfully migrated data to {db_path}!")

if __name__ == '__main__':
    seed_domains()
