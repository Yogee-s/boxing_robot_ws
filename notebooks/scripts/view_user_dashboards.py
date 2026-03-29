"""Render styled profile cards for each demo user."""
import sqlite3
import os
from IPython.display import HTML, display

demo_users = ['alex', 'maria', 'jake']
cards_html = ""

for username in demo_users:
    db_path = f'data/users/{username}/boxbunny.db'
    if not os.path.exists(db_path):
        print(f"Database not found for {username} "
              "-- run Section 4 (Seed Demo Data) first")
        continue

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    xp = dict(conn.execute(
        "SELECT * FROM user_xp WHERE id=1").fetchone())
    streak = dict(conn.execute(
        "SELECT * FROM streaks WHERE id=1").fetchone())
    sessions = conn.execute(
        "SELECT COUNT(*) as c FROM training_sessions").fetchone()['c']
    achievements = conn.execute(
        "SELECT COUNT(*) as c FROM achievements").fetchone()['c']
    records = [dict(r) for r in conn.execute(
        "SELECT * FROM personal_records").fetchall()]
    conn.close()

    main = sqlite3.connect('data/boxbunny_main.db')
    main.row_factory = sqlite3.Row
    user = dict(main.execute(
        "SELECT * FROM users WHERE username=?", (username,)).fetchone())
    main.close()

    records_html = ""
    for r in records[:4]:
        records_html += (
            f'<div style="background:#111;padding:6px 10px;'
            f'border-radius:6px;margin:2px 0;font-size:12px">'
            f'<span style="color:#FFC107">'
            f'{r["metric"].replace("_"," ").title()}</span>: '
            f'<b style="color:white">{r["value"]}</b>'
            f'</div>'
        )

    xp_pct = min(100, int(xp.get('level_progress', 0.5) * 100))

    cards_html += f"""
    <div style="background:#1A1A1A;padding:20px;border-radius:12px;
                margin:12px 0;font-family:sans-serif;color:white;
                border-left:4px solid #00E676">
        <div style="display:flex;justify-content:space-between;
                    align-items:center">
            <h3 style="color:#00E676;margin:0">
                {user['display_name']}</h3>
            <span style="background:#00E676;color:#0D0D0D;
                         padding:4px 12px;border-radius:12px;
                         font-weight:bold;font-size:13px">
                {xp['current_rank']}</span>
        </div>
        <p style="color:#9E9E9E;margin:4px 0 0 0;font-size:13px">
            {user.get('gender','?').title()}, Age {user.get('age','?')} |
            {user.get('height_cm','?')}cm, {user.get('weight_kg','?')}kg |
            {user['level'].title()} |
            {user.get('stance','orthodox').title()}
        </p>
        <div style="display:flex;gap:12px;margin-top:16px">
            <div style="background:#0D0D0D;padding:14px;
                        border-radius:8px;flex:1;text-align:center">
                <div style="font-size:28px;font-weight:bold;
                            color:#00E676">{xp['total_xp']}</div>
                <div style="color:#9E9E9E;font-size:11px">TOTAL XP</div>
            </div>
            <div style="background:#0D0D0D;padding:14px;
                        border-radius:8px;flex:1;text-align:center">
                <div style="font-size:28px;font-weight:bold">
                    {sessions}</div>
                <div style="color:#9E9E9E;font-size:11px">SESSIONS</div>
            </div>
            <div style="background:#0D0D0D;padding:14px;
                        border-radius:8px;flex:1;text-align:center">
                <div style="font-size:28px;font-weight:bold;
                            color:#FF5722">{streak['current_streak']}</div>
                <div style="color:#9E9E9E;font-size:11px">DAY STREAK</div>
            </div>
            <div style="background:#0D0D0D;padding:14px;
                        border-radius:8px;flex:1;text-align:center">
                <div style="font-size:28px;font-weight:bold;
                            color:#FFC107">{achievements}</div>
                <div style="color:#9E9E9E;font-size:11px">
                    ACHIEVEMENTS</div>
            </div>
        </div>
        <div style="margin-top:12px">
            <div style="color:#9E9E9E;font-size:11px;margin-bottom:4px">
                PERSONAL RECORDS</div>
            <div style="display:flex;gap:6px;flex-wrap:wrap">
                {records_html}</div>
        </div>
    </div>
    """

display(HTML(f"""
<div style="max-width:700px">
    <h2 style="color:#E0E0E0;font-family:sans-serif;margin-bottom:4px">
        Demo User Profiles</h2>
    <p style="color:#9E9E9E;font-family:sans-serif;font-size:13px;
              margin-top:0">
        Data pulled live from per-user SQLite databases in data/users/
    </p>
    {cards_html}
</div>
"""))
