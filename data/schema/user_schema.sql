-- BoxBunny Per-User Database Schema
-- Per-user database: data/users/<username>/boxbunny.db

CREATE TABLE IF NOT EXISTS training_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    mode TEXT NOT NULL,
    difficulty TEXT NOT NULL DEFAULT 'beginner',
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at TEXT,
    is_complete INTEGER NOT NULL DEFAULT 0,
    rounds_completed INTEGER NOT NULL DEFAULT 0,
    rounds_total INTEGER NOT NULL DEFAULT 0,
    work_time_sec INTEGER NOT NULL DEFAULT 180,
    rest_time_sec INTEGER NOT NULL DEFAULT 60,
    config_json TEXT NOT NULL DEFAULT '{}',
    summary_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS session_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    event_type TEXT NOT NULL,
    data_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS combo_progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    combo_id TEXT NOT NULL,
    combo_name TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    best_accuracy REAL NOT NULL DEFAULT 0.0,
    avg_accuracy REAL NOT NULL DEFAULT 0.0,
    mastered INTEGER NOT NULL DEFAULT 0,
    last_attempted TEXT,
    UNIQUE(combo_id)
);

CREATE TABLE IF NOT EXISTS power_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tested_at TEXT NOT NULL DEFAULT (datetime('now')),
    peak_force REAL NOT NULL DEFAULT 0.0,
    avg_force REAL NOT NULL DEFAULT 0.0,
    punch_count INTEGER NOT NULL DEFAULT 0,
    results_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS stamina_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tested_at TEXT NOT NULL DEFAULT (datetime('now')),
    duration_sec INTEGER NOT NULL DEFAULT 120,
    total_punches INTEGER NOT NULL DEFAULT 0,
    punches_per_minute REAL NOT NULL DEFAULT 0.0,
    fatigue_index REAL NOT NULL DEFAULT 0.0,
    results_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS reaction_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tested_at TEXT NOT NULL DEFAULT (datetime('now')),
    num_trials INTEGER NOT NULL DEFAULT 10,
    avg_reaction_ms REAL NOT NULL DEFAULT 0.0,
    best_reaction_ms REAL NOT NULL DEFAULT 0.0,
    worst_reaction_ms REAL NOT NULL DEFAULT 0.0,
    tier TEXT NOT NULL DEFAULT 'average',
    results_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS sparring_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    style TEXT NOT NULL DEFAULT 'boxer',
    difficulty TEXT NOT NULL DEFAULT 'medium',
    rounds_completed INTEGER NOT NULL DEFAULT 0,
    user_punches INTEGER NOT NULL DEFAULT 0,
    robot_punches_thrown INTEGER NOT NULL DEFAULT 0,
    robot_punches_landed INTEGER NOT NULL DEFAULT 0,
    defense_rate REAL NOT NULL DEFAULT 0.0,
    punch_distribution_json TEXT NOT NULL DEFAULT '{}',
    defense_breakdown_json TEXT NOT NULL DEFAULT '{}',
    completed_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sparring_weakness_profile (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    punch_type TEXT NOT NULL,
    defense_success_rate REAL NOT NULL DEFAULT 0.5,
    exposure_count INTEGER NOT NULL DEFAULT 0,
    last_updated TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(punch_type)
);

-- Gamification tables
CREATE TABLE IF NOT EXISTS user_xp (
    id INTEGER PRIMARY KEY CHECK(id = 1),
    total_xp INTEGER NOT NULL DEFAULT 0,
    current_rank TEXT NOT NULL DEFAULT 'Novice',
    rank_history_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS personal_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_type TEXT UNIQUE NOT NULL,
    value REAL NOT NULL DEFAULT 0.0,
    achieved_at TEXT NOT NULL DEFAULT (datetime('now')),
    previous_value REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS achievements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    achievement_id TEXT UNIQUE NOT NULL,
    unlocked_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS streaks (
    id INTEGER PRIMARY KEY CHECK(id = 1),
    current_streak INTEGER NOT NULL DEFAULT 0,
    longest_streak INTEGER NOT NULL DEFAULT 0,
    last_training_date TEXT,
    weekly_goal INTEGER NOT NULL DEFAULT 3,
    weekly_progress INTEGER NOT NULL DEFAULT 0,
    week_start_date TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_training_sessions_id ON training_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_training_sessions_mode ON training_sessions(mode);
CREATE INDEX IF NOT EXISTS idx_session_events_session ON session_events(session_id);
CREATE INDEX IF NOT EXISTS idx_session_events_type ON session_events(event_type);
CREATE INDEX IF NOT EXISTS idx_combo_progress_combo ON combo_progress(combo_id);

-- Initialize singleton rows
INSERT OR IGNORE INTO user_xp (id, total_xp, current_rank, rank_history_json)
VALUES (1, 0, 'Novice', '[]');

INSERT OR IGNORE INTO streaks (id, current_streak, longest_streak, weekly_goal, weekly_progress)
VALUES (1, 0, 0, 3, 0);
