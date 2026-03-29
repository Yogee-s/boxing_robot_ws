"""Inspect a per-user BoxBunny database."""
import sqlite3
import sys

USERNAME = sys.argv[1] if len(sys.argv) > 1 else "maria"

db_path = f'data/users/{USERNAME}/boxbunny.db'
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

print(f"=== {USERNAME}'s Training Sessions (last 10) ===")
for row in conn.execute(
    "SELECT session_id, mode, difficulty, rounds_completed, is_complete "
    "FROM training_sessions ORDER BY started_at DESC LIMIT 10"
):
    print(f"  {dict(row)}")

print(f"\n=== {USERNAME}'s XP & Rank ===")
for row in conn.execute("SELECT * FROM user_xp"):
    print(f"  {dict(row)}")

print(f"\n=== {USERNAME}'s Streak ===")
for row in conn.execute("SELECT * FROM streaks"):
    print(f"  {dict(row)}")

print(f"\n=== {USERNAME}'s Achievements ===")
for row in conn.execute("SELECT * FROM achievements"):
    print(f"  {dict(row)}")

print(f"\n=== {USERNAME}'s Personal Records ===")
for row in conn.execute("SELECT * FROM personal_records"):
    print(f"  {dict(row)}")

conn.close()
