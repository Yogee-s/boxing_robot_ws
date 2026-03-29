#!/usr/bin/env python3
"""Demo data seeder for BoxBunny dashboard.

Creates realistic demo users with full training history, performance data,
gamification progress, and presets. Run from project root:

    python3 tools/demo_data_seeder.py          # seed data
    python3 tools/demo_data_seeder.py --clean   # wipe and reseed
"""
import argparse, json, os, random, secrets, shutil, sys
from datetime import datetime, timedelta

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "src", "boxbunny_dashboard"))
from boxbunny_dashboard.db.manager import DatabaseManager

DATA_DIR = os.path.join(WS, "data")
PUNCHES = ["jab", "cross", "hook_lead", "hook_rear", "uppercut_lead", "uppercut_rear"]
PADS = ["head_left", "head_right", "body_left", "body_right"]
FORCES = ["light", "medium", "hard"]
NOW = datetime.utcnow()

# ── Helpers ──────────────────────────────────────────────────────────────────

ts = lambda dt: dt.isoformat(timespec="seconds")

PUNCH_W = {"beginner": [.40,.30,.12,.10,.04,.04],
            "intermediate": [.28,.25,.16,.14,.09,.08],
            "advanced": [.20,.20,.17,.17,.13,.13]}
FORCE_W = {"beginner": [.50,.35,.15], "intermediate": [.25,.45,.30], "advanced": [.15,.35,.50]}

def spread_dates(count, span_days):
    dates = sorted(NOW - timedelta(days=random.randint(0, span_days)) for _ in range(count))
    out = []
    for d in dates:
        h = random.randint(6, 10) if random.random() < 0.5 else random.randint(17, 21)
        out.append(d.replace(hour=h, minute=random.randint(0, 59), second=random.randint(0, 59)))
    return out

def gen_punches(total, level):
    pw, fw = PUNCH_W[level], FORCE_W[level]
    p = {k: 0 for k in PUNCHES}; f = {k: 0 for k in FORCES}; pad = {k: 0 for k in PADS}
    for _ in range(total):
        p[random.choices(PUNCHES, pw)[0]] += 1
        f[random.choices(FORCES, fw)[0]] += 1
        pad[random.choice(PADS)] += 1
    return p, f, pad

def gen_session(db, user, dt, mode, diff, rounds, work, rest, level):
    total = max(40, int(random.uniform(.7, 1.3) * rounds * work / 3 *
                        {"beginner": .6, "intermediate": .85, "advanced": 1.1}[level]))
    dur = (rounds * (work + rest)) / 60.0
    p, f, pad = gen_punches(total, level)
    summary = {"total_punches": total, "duration_minutes": dur,
               "punch_distribution": p, "force_distribution": f, "pad_distribution": pad,
               "avg_punch_speed_ms": random.randint(180, 400),
               "calories_est": int(dur * random.uniform(8, 14)), "mode": mode}
    sid = secrets.token_urlsafe(12)
    db.save_training_session(user, {
        "session_id": sid, "mode": mode, "difficulty": diff,
        "started_at": ts(dt), "ended_at": ts(dt + timedelta(minutes=int(dur) + 3)),
        "is_complete": True, "rounds_completed": rounds, "rounds_total": rounds,
        "work_time_sec": work, "rest_time_sec": rest, "summary": summary})
    t = dt.timestamp()
    db.save_session_event(user, sid, t, "session_start", {"mode": mode})
    for r in range(rounds):
        rt = t + r * (work + rest)
        db.save_session_event(user, sid, rt, "round_start", {"round": r + 1})
        db.save_session_event(user, sid, rt + work, "round_end", {"round": r + 1})
    db.save_session_event(user, sid, t + rounds * (work + rest), "session_end", summary)
    return sid

def gen_sparring(db, user, sid, dt, diff, level):
    rate = {"advanced": (.70,.90), "intermediate": (.55,.75), "beginner": (.35,.55)}[level]
    dr = random.uniform(*rate); thrown = random.randint(40, 140)
    p, _, _ = gen_punches(random.randint(60, 200), level)
    conn = db._get_user_conn(user)
    try:
        conn.execute(
            """INSERT INTO sparring_sessions (session_id,style,difficulty,rounds_completed,
               user_punches,robot_punches_thrown,robot_punches_landed,defense_rate,
               punch_distribution_json,defense_breakdown_json,completed_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (sid, random.choice(["boxer","brawler","counter_puncher","swarmer"]), diff,
             random.randint(2, 5), sum(p.values()), thrown, int(thrown * (1 - dr)),
             round(dr, 3), json.dumps(p),
             json.dumps({"slip": random.randint(5, 30), "block": random.randint(5, 30),
                         "parry": random.randint(2, 15)}), ts(dt)))
        conn.commit()
    finally:
        conn.close()

def set_xp(db, user, xp, rank):
    ranks = [("Novice",0),("Contender",500),("Fighter",1500),("Warrior",4000),
             ("Champion",10000),("Elite",25000)]
    hist = [{"rank": r, "achieved_at": NOW.isoformat(), "xp": t}
            for r, t in ranks if t > 0 and xp >= t and ranks.index((r, t)) <= [i for i,(rr,_) in enumerate(ranks) if rr == rank][0]]
    conn = db._get_user_conn(user)
    try:
        conn.execute("UPDATE user_xp SET total_xp=?, current_rank=?, rank_history_json=? WHERE id=1",
                     (xp, rank, json.dumps(hist)))
        conn.commit()
    finally:
        conn.close()

def set_streak(db, user, cur, longest, goal, progress):
    last = (NOW.date() - timedelta(days=0 if cur > 0 else 1)).isoformat()
    conn = db._get_user_conn(user)
    try:
        conn.execute("UPDATE streaks SET current_streak=?,longest_streak=?,last_training_date=?,weekly_goal=?,weekly_progress=? WHERE id=1",
                     (cur, longest, last, goal, progress))
        conn.commit()
    finally:
        conn.close()

def set_proficiency(db, user, answers):
    conn = db._get_main_conn()
    try:
        conn.execute("UPDATE users SET proficiency_answers_json=? WHERE username=?",
                     (json.dumps(answers), user))
        conn.commit()
    finally:
        conn.close()

def add_coaching(db, coach_id, name, n_part, dt, preset_id=None):
    conn = db._get_main_conn()
    try:
        cur = conn.execute(
            """INSERT INTO coaching_sessions (coach_user_id,station_config_preset_id,
               started_at,ended_at,total_participants,notes) VALUES (?,?,?,?,?,?)""",
            (coach_id, preset_id, ts(dt), ts(dt + timedelta(hours=1)), n_part, name))
        cs_id = cur.lastrowid
        for i in range(1, n_part + 1):
            conn.execute(
                "INSERT INTO coaching_participants (coaching_session_id,participant_number,participant_name,session_data_json) VALUES (?,?,?,?)",
                (cs_id, i, f"Participant {i}", json.dumps({
                    "punch_count": random.randint(80, 250),
                    "punch_distribution": {k: random.randint(5, 50) for k in PUNCHES},
                    "reaction_time_ms": random.randint(200, 400),
                    "fatigue_index": round(random.uniform(0.1, 0.6), 2)})))
        conn.commit()
    finally:
        conn.close()

def make_presets(db, uid, defs, fav_count=0):
    for i, (name, ptype, cfg, tags) in enumerate(defs):
        pid = db.create_preset(uid, name, ptype, json.dumps(cfg), tags=tags)
        if i < fav_count:
            db.update_preset(pid, is_favorite=1)

def gen_sessions(db, user, dates, modes, diff, level, work=180, rest=60):
    random.shuffle(modes)
    for i, dt in enumerate(dates):
        rounds = random.randint(2 if level == "beginner" else 3, 5 if level != "advanced" else 8)
        sid = gen_session(db, user, dt, modes[i], diff, rounds, work, rest, level)
        if modes[i] == "sparring":
            gen_sparring(db, user, sid, dt, diff, level)

# ── User Seeders ─────────────────────────────────────────────────────────────

def seed_alex(db):
    print("  Creating Alex (beginner) — M, 22, 175cm, 72kg...")
    uid = db.create_user("alex", "boxing123", "Alex Chen", "individual", "beginner",
                         age=22, gender="male", height_cm=175.0, weight_kg=72.0,
                         reach_cm=178.0, stance="orthodox")
    if not uid: print("    Exists, skip."); return
    db.set_pattern(uid, [0, 1, 2, 5, 8])
    set_proficiency(db, "alex", {"boxing_experience": "no", "goal": "fitness", "intensity": "light"})
    gen_sessions(db, "alex", spread_dates(8, 14),
                 ["training"]*5 + ["free"]*3, "beginner", "beginner")
    db.save_stamina_test("alex", {"duration_sec": 120, "total_punches": 95,
        "punches_per_minute": 47.5, "fatigue_index": 0.35, "results": []})
    db.save_reaction_test("alex", {"num_trials": 10, "avg_reaction_ms": 340,
        "best_reaction_ms": 280, "worst_reaction_ms": 420, "tier": "Average", "results": []})
    set_xp(db, "alex", 350, "Novice")
    set_streak(db, "alex", 3, 5, 3, 2)
    for a in ["first_blood", "century", "well_rounded"]:
        db.unlock_achievement("alex", a)
    r = {"rounds": 2, "work_sec": 120, "rest_sec": 60, "difficulty": "beginner"}
    make_presets(db, uid, [
        ("Quick Warmup", "training", r, ""),
        ("Jab Practice", "training", {**r, "rounds": 3, "work_sec": 180}, ""),
    ], fav_count=1)
    print("    8 sessions, 2 tests, 3 achievements, 2 presets.")

def seed_maria(db):
    print("  Creating Maria (intermediate) — F, 28, 165cm, 58kg...")
    uid = db.create_user("maria", "boxing123", "Maria Santos", "individual", "intermediate",
                         age=28, gender="female", height_cm=165.0, weight_kg=58.0,
                         reach_cm=163.0, stance="orthodox")
    if not uid: print("    Exists, skip."); return
    db.set_pattern(uid, [0, 3, 6, 7, 8, 5, 2])
    set_proficiency(db, "maria", {"boxing_experience": "some", "goal": "skill", "intensity": "moderate"})
    gen_sessions(db, "maria", spread_dates(35, 90),
                 ["training"]*15 + ["sparring"]*8 + ["free"]*7 + ["performance"]*5,
                 "intermediate", "intermediate")
    for _ in range(2):
        db.save_power_test("maria", {"peak_force": round(random.uniform(600,900),1),
            "avg_force": round(random.uniform(350,550),1), "punch_count": random.randint(15,25), "results": []})
    for _ in range(2):
        db.save_stamina_test("maria", {"duration_sec": 180, "total_punches": random.randint(180,260),
            "punches_per_minute": round(random.uniform(55,80),1),
            "fatigue_index": round(random.uniform(.2,.4),2), "results": []})
    db.save_reaction_test("maria", {"num_trials": 15, "avg_reaction_ms": 280,
        "best_reaction_ms": 220, "worst_reaction_ms": 360, "tier": "Fast", "results": []})
    set_xp(db, "maria", 2800, "Fighter")
    set_streak(db, "maria", 12, 18, 4, 3)
    for a in ["first_blood","century","fury","iron_chin","weekly_warrior","well_rounded","speed_demon","combo_breaker"]:
        db.unlock_achievement("maria", a)
    c = {"difficulty": "intermediate", "rounds": 4, "work_sec": 180, "rest_sec": 60}
    make_presets(db, uid, [
        ("Power Combo","training",c,""), ("Speed Drill","training",c,""),
        ("Cardio Blast","training",c,""), ("Sparring Prep","sparring",c,""),
        ("Cool Down","free",c,""),
    ], fav_count=2)
    for rt, v in [("max_punch_speed_ms",165.0),("max_session_punches",248.0),("best_reaction_ms",220.0)]:
        db.check_personal_record("maria", rt, v)
    print("    35 sessions, 5 tests, 8 achievements, 5 presets.")

def seed_jake(db):
    print("  Creating Jake (advanced) — M, 31, 183cm, 82kg...")
    uid = db.create_user("jake", "boxing123", "Jake Thompson", "individual", "advanced",
                         age=31, gender="male", height_cm=183.0, weight_kg=82.0,
                         reach_cm=188.0, stance="orthodox")
    if not uid: print("    Exists, skip."); return
    db.set_pattern(uid, [0, 4, 8, 6, 2, 1, 3, 5, 7])
    set_proficiency(db, "jake", {"boxing_experience": "yes", "goal": "competition", "intensity": "hard"})
    gen_sessions(db, "jake", spread_dates(120, 240),
                 ["training"]*45 + ["sparring"]*40 + ["free"]*20 + ["performance"]*15,
                 "advanced", "advanced", work=180, rest=45)
    for _ in range(5):
        db.save_power_test("jake", {"peak_force": round(random.uniform(900,1400),1),
            "avg_force": round(random.uniform(600,900),1), "punch_count": random.randint(20,35), "results": []})
    for _ in range(5):
        db.save_stamina_test("jake", {"duration_sec": 300, "total_punches": random.randint(350,550),
            "punches_per_minute": round(random.uniform(70,110),1),
            "fatigue_index": round(random.uniform(.1,.25),2), "results": []})
    for _ in range(5):
        db.save_reaction_test("jake", {"num_trials": 20,
            "avg_reaction_ms": round(random.uniform(210,235),1),
            "best_reaction_ms": round(random.uniform(170,200),1),
            "worst_reaction_ms": round(random.uniform(260,310),1), "tier": "Lightning", "results": []})
    set_xp(db, "jake", 12000, "Champion")
    set_streak(db, "jake", 28, 45, 5, 4)
    for a in ["first_blood","century","fury","iron_chin","weekly_warrior","well_rounded",
              "speed_demon","combo_breaker","power_surge","marathon","untouchable",
              "perfect_round","streak_master","shadow_king","ring_general"]:
        db.unlock_achievement("jake", a)
    c = {"difficulty": "advanced", "rounds": 6, "work_sec": 180, "rest_sec": 45}
    make_presets(db, uid, [
        ("Championship Prep","sparring",c,""), ("Power Rounds","training",c,""),
        ("Speed Ladder","training",c,""), ("Endurance Circuit","circuit",c,""),
        ("Sparring: Counter","sparring",c,""), ("Sparring: Pressure","sparring",c,""),
        ("Active Recovery","free",c,""), ("Fight Simulation","sparring",c,""),
    ], fav_count=3)
    for rt, v in [("max_punch_speed_ms",135.0),("max_session_punches",482.0),
                  ("best_reaction_ms",172.0),("max_peak_force",1380.0),("longest_streak",45.0)]:
        db.check_personal_record("jake", rt, v)
    print("    120 sessions, 15 tests, 15 achievements, 8 presets.")

def seed_sarah(db):
    print("  Creating Coach Sarah — F, 35, 170cm, 65kg...")
    uid = db.create_user("sarah", "coaching123", "Coach Sarah", "coach", "advanced",
                         age=35, gender="female", height_cm=170.0, weight_kg=65.0,
                         reach_cm=168.0, stance="orthodox")
    if not uid: print("    Exists, skip."); return
    db.set_pattern(uid, [0, 1, 2, 5, 4, 3, 6, 7, 8])
    c = {"difficulty": "intermediate", "rounds": 4, "work_sec": 180, "rest_sec": 60}
    pids = []
    for name, tag in [("Warmup Routine","warmup"),("Cardio Boxing","cardio"),
                      ("Technique Focus","technique"),("Cool-Down Stretch","cool-down")]:
        pids.append(db.create_preset(uid, name, "circuit", json.dumps(c), tags=tag))
    for name, n, days, pi in [("Beginner Circuit",6,5,0),("Cardio Boxing",8,3,1),("Advanced Drills",4,1,2)]:
        add_coaching(db, uid, name, n, NOW - timedelta(days=days), pids[pi])
    print("    3 coaching sessions (18 participants), 4 presets.")

def seed_guest(db):
    print("  Creating guest session...")
    token = db.create_guest_session(ttl_days=7)
    print(f"    Guest token: {token}")

# ── Main ─────────────────────────────────────────────────────────────────────

def clean(data_dir):
    main_db = os.path.join(data_dir, "boxbunny_main.db")
    users_dir = os.path.join(data_dir, "users")
    for f in [main_db, main_db + "-wal", main_db + "-shm"]:
        if os.path.exists(f):
            os.remove(f); print(f"  Removed {f}")
    if os.path.isdir(users_dir):
        shutil.rmtree(users_dir); print(f"  Removed {users_dir}/")

def main():
    ap = argparse.ArgumentParser(description="Seed BoxBunny demo data")
    ap.add_argument("--clean", action="store_true", help="Wipe databases before seeding")
    args = ap.parse_args()
    print("BoxBunny Demo Data Seeder")
    print(f"Data directory: {DATA_DIR}")
    if args.clean:
        print("\nCleaning existing data..."); clean(DATA_DIR)
    print("\nInitializing database manager...")
    db = DatabaseManager(DATA_DIR)
    print("\nSeeding demo users:")
    seed_alex(db); seed_maria(db); seed_jake(db); seed_sarah(db); seed_guest(db)
    print("\nDone! Credentials:")
    print("  alex   / boxing123    - Beginner (8 sessions)")
    print("  maria  / boxing123    - Intermediate (35 sessions)")
    print("  jake   / boxing123    - Advanced (120 sessions)")
    print("  sarah  / coaching123  - Coach (3 coaching sessions)")

if __name__ == "__main__":
    main()
