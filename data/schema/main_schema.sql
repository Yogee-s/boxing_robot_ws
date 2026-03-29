-- BoxBunny Main Database Schema
-- Shared database: data/boxbunny_main.db

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    pattern_hash TEXT,
    display_name TEXT NOT NULL,
    user_type TEXT NOT NULL DEFAULT 'individual' CHECK(user_type IN ('individual', 'coach')),
    level TEXT NOT NULL DEFAULT 'beginner' CHECK(level IN ('beginner', 'intermediate', 'advanced')),
    age INTEGER,
    gender TEXT CHECK(gender IN ('male', 'female', 'other', NULL)),
    height_cm REAL,
    weight_kg REAL,
    reach_cm REAL,
    stance TEXT DEFAULT 'orthodox' CHECK(stance IN ('orthodox', 'southpaw')),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_login TEXT,
    settings_json TEXT NOT NULL DEFAULT '{}',
    proficiency_answers_json TEXT
);

CREATE TABLE IF NOT EXISTS auth_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    session_token TEXT UNIQUE NOT NULL,
    device_type TEXT NOT NULL CHECK(device_type IN ('phone', 'robot')),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS guest_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guest_session_token TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    claimed_by_user_id INTEGER,
    claimed_at TEXT,
    expires_at TEXT NOT NULL,
    FOREIGN KEY (claimed_by_user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS presets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    preset_type TEXT NOT NULL CHECK(preset_type IN ('training', 'sparring', 'performance', 'circuit', 'free')),
    config_json TEXT NOT NULL DEFAULT '{}',
    is_favorite INTEGER NOT NULL DEFAULT 0,
    tags TEXT DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    use_count INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS coaching_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    coach_user_id INTEGER NOT NULL,
    station_config_preset_id INTEGER,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at TEXT,
    total_participants INTEGER NOT NULL DEFAULT 0,
    notes TEXT DEFAULT '',
    FOREIGN KEY (coach_user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (station_config_preset_id) REFERENCES presets(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS coaching_participants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    coaching_session_id INTEGER NOT NULL,
    participant_number INTEGER NOT NULL,
    participant_name TEXT DEFAULT '',
    session_data_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (coaching_session_id) REFERENCES coaching_sessions(id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_user_type ON users(user_type);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_token ON auth_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON auth_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_guest_sessions_token ON guest_sessions(guest_session_token);
CREATE INDEX IF NOT EXISTS idx_presets_user ON presets(user_id);
CREATE INDEX IF NOT EXISTS idx_coaching_participants_session ON coaching_participants(coaching_session_id);
