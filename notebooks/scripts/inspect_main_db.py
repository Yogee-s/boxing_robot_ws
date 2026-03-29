"""Inspect the main BoxBunny database."""
import sqlite3

conn = sqlite3.connect('data/boxbunny_main.db')
conn.row_factory = sqlite3.Row

print("=== Users ===")
for row in conn.execute(
    "SELECT id, username, display_name, user_type, level, last_login "
    "FROM users"
):
    print(f"  {dict(row)}")

print("\n=== Presets ===")
for row in conn.execute(
    "SELECT user_id, name, preset_type, is_favorite, use_count "
    "FROM presets LIMIT 10"
):
    print(f"  {dict(row)}")

print("\n=== Coaching Sessions ===")
for row in conn.execute("SELECT * FROM coaching_sessions"):
    print(f"  {dict(row)}")

print("\n=== Guest Sessions ===")
for row in conn.execute("SELECT * FROM guest_sessions"):
    print(f"  {dict(row)}")

conn.close()
