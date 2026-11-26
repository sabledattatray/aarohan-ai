import sqlite3
from pathlib import Path

DB_PATH = Path(r"D:\Datta Sable\AAROHAN\data\memory\aarohan.sqlite")

def repair():
    print(f"Connecting to {DB_PATH}...")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # 1. Inspect before repair
    row_before = conn.execute("SELECT * FROM learned_rules WHERE id = 'rule_e78d030b7beb'").fetchone()
    if not row_before:
        raise RuntimeError("Rule rule_e78d030b7beb not found!")

    print("BEFORE REPAIR:")
    print(f"  rule_key:  {row_before['rule_key']}")
    print(f"  value:     {row_before['value']}")
    print(f"  rule_text: '{row_before['rule_text']}'")
    print(f"  version:   {row_before['version']}")

    # 2. Atomic transaction
    new_rule_text = "LEVEL-A cases have response deadline of 12 hours."

    with conn:
        conn.execute("""
            UPDATE learned_rules
            SET rule_text = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = 'rule_e78d030b7beb'
        """, (new_rule_text,))

        # Synchronize FTS table
        conn.execute("DELETE FROM learned_rules_fts WHERE rule_id = 'rule_e78d030b7beb'")
        conn.execute("""
            INSERT INTO learned_rules_fts (rule_id, namespace, rule_key, subject, predicate, value, conditions, rule_text)
            SELECT id, namespace, rule_key, subject, predicate, value, conditions, rule_text
            FROM learned_rules WHERE id = 'rule_e78d030b7beb'
        """)

    print("\nATOMIC TRANSACTION COMMITTED SUCCESSFULLY.")

    # 3. Inspect after repair
    row_after = conn.execute("SELECT * FROM learned_rules WHERE id = 'rule_e78d030b7beb'").fetchone()
    fts_after = conn.execute("SELECT * FROM learned_rules_fts WHERE rule_id = 'rule_e78d030b7beb'").fetchone()

    print("AFTER REPAIR:")
    print(f"  rule_key:  {row_after['rule_key']}")
    print(f"  value:     {row_after['value']}")
    print(f"  rule_text: '{row_after['rule_text']}'")
    print(f"  FTS text:  '{fts_after['rule_text']}'")
    print(f"  version:   {row_after['version']}")

    assert row_after['rule_text'] == new_rule_text, "rule_text does not match!"
    assert fts_after['rule_text'] == new_rule_text, "FTS rule_text does not match!"
    print("\nREPAIR VERIFIED 100% IN DATABASE.")
    conn.close()

if __name__ == "__main__":
    repair()
