import sqlite3
import json

def inspect_db():
    conn = sqlite3.connect('data/atlas.db')
    cursor = conn.cursor()
    
    print("=== First 5 Models ===")
    cursor.execute("SELECT id, name, provider FROM models LIMIT 5")
    for row in cursor.fetchall():
        print(f"ID: {row[0]:<3} | Name: {row[1]:<20} | Provider: {row[2]}")
        
    print("\n=== First 5 Domain Examples ===")
    cursor.execute("""
        SELECT d.name, e.example_text 
        FROM domain_examples e 
        JOIN domains d ON e.domain_id = d.id 
        LIMIT 5
    """)
    for row in cursor.fetchall():
        print(f"Domain: {row[0]:<10} | Example: {row[1]}")
        
    print("\n=== GPT-4o Metrics (JSON) ===")
    cursor.execute("""
        SELECT m.name, metrics.performance_json 
        FROM models m 
        JOIN model_metrics metrics ON m.id = metrics.model_id 
        WHERE m.name = 'GPT-4o'
    """)
    row = cursor.fetchone()
    if row:
        print(f"Model: {row[0]}")
        # Pretty-print the JSON
        parsed_json = json.loads(row[1])
        print(json.dumps(parsed_json, indent=2))
        
    conn.close()

if __name__ == "__main__":
    inspect_db()
