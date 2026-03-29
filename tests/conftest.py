"""Shared test fixtures for BoxBunny tests.

All fixtures use mock data -- no hardware, ROS, camera, or IMU required.
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest

# Add source paths for imports
WS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS_ROOT / "src" / "boxbunny_dashboard"))
sys.path.insert(0, str(WS_ROOT / "src" / "boxbunny_core"))
sys.path.insert(0, str(WS_ROOT / "action_prediction"))


# ── Database fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory with schema files."""
    schema_dir = tmp_path / "schema"
    schema_dir.mkdir()
    src_schema = WS_ROOT / "data" / "schema"
    for sql_file in src_schema.glob("*.sql"):
        (schema_dir / sql_file.name).write_text(sql_file.read_text())
    return tmp_path


@pytest.fixture
def db_manager(tmp_data_dir):
    """Database manager with temporary databases."""
    from boxbunny_dashboard.db.manager import DatabaseManager
    return DatabaseManager(str(tmp_data_dir))


@pytest.fixture
def sample_user(db_manager):
    """Create and return a sample user."""
    user_id = db_manager.create_user(
        username="testuser",
        password="testpass123",
        display_name="Test User",
        user_type="individual",
        level="beginner",
    )
    return {"id": user_id, "username": "testuser", "display_name": "Test User"}


# ── User profile fixtures ───────────────────────────────────────────────────

@pytest.fixture
def beginner_profile():
    """A beginner user profile for testing."""
    return {
        "username": "beginner_bob",
        "display_name": "Beginner Bob",
        "level": "beginner",
        "total_xp": 120,
        "current_rank": "Novice",
        "sessions_completed": 4,
        "preferred_combos": ["1-2", "1-1-2"],
    }


@pytest.fixture
def intermediate_profile():
    """An intermediate user profile for testing."""
    return {
        "username": "inter_ida",
        "display_name": "Intermediate Ida",
        "level": "intermediate",
        "total_xp": 2800,
        "current_rank": "Fighter",
        "sessions_completed": 45,
        "preferred_combos": ["1-2-3", "1-2-5", "3b-3"],
    }


@pytest.fixture
def coach_profile():
    """A coach user profile for testing."""
    return {
        "username": "coach_charlie",
        "display_name": "Coach Charlie",
        "level": "advanced",
        "user_type": "coach",
        "total_xp": 15000,
        "current_rank": "Champion",
    }


# ── Session data fixtures ───────────────────────────────────────────────────

@pytest.fixture
def sample_session_data():
    """Sample training session data."""
    return {
        "session_id": "test_session_001",
        "mode": "training",
        "difficulty": "beginner",
        "started_at": "2026-03-29T10:00:00",
        "ended_at": "2026-03-29T10:15:00",
        "is_complete": True,
        "rounds_completed": 3,
        "rounds_total": 3,
        "work_time_sec": 180,
        "rest_time_sec": 60,
        "config": {"combo": "jab-cross-hook", "speed": "medium"},
        "summary": {
            "total_punches": 87,
            "punch_distribution": {"jab": 35, "cross": 30, "left_hook": 22},
            "defense_rate": 0.75,
            "avg_depth": 1.5,
        },
    }


@pytest.fixture
def sample_sparring_session():
    """Sample sparring session data."""
    return {
        "session_id": "sparring_001",
        "mode": "sparring",
        "difficulty": "intermediate",
        "started_at": "2026-03-29T14:00:00",
        "ended_at": "2026-03-29T14:20:00",
        "is_complete": True,
        "rounds_completed": 4,
        "rounds_total": 4,
        "work_time_sec": 180,
        "rest_time_sec": 60,
        "config": {"style": "boxer", "speed": "medium", "robot_difficulty": "medium"},
        "summary": {
            "total_punches": 210,
            "robot_punches_thrown": 32,
            "robot_punches_landed": 8,
            "defense_rate": 0.75,
        },
    }


# ── Punch event fixtures ────────────────────────────────────────────────────

@pytest.fixture
def sample_punch_events():
    """Sample confirmed punch events."""
    return [
        {"type": "jab", "pad": "centre", "force": 0.66, "cv_conf": 0.85, "ts": 1.0},
        {"type": "cross", "pad": "centre", "force": 1.0, "cv_conf": 0.92, "ts": 1.5},
        {"type": "left_hook", "pad": "right", "force": 0.33, "cv_conf": 0.78, "ts": 2.1},
        {"type": "jab", "pad": "centre", "force": 0.66, "cv_conf": 0.88, "ts": 3.0},
        {"type": "cross", "pad": "left", "force": 1.0, "cv_conf": 0.91, "ts": 3.4},
    ]


@pytest.fixture
def sample_cv_detections():
    """Sample CV detection events for fusion testing."""
    base_ts = time.time()
    return [
        {"timestamp": base_ts + 0.0, "punch_type": "jab", "confidence": 0.88, "raw_class": "jab"},
        {"timestamp": base_ts + 0.5, "punch_type": "cross", "confidence": 0.91, "raw_class": "cross"},
        {"timestamp": base_ts + 1.2, "punch_type": "left_hook", "confidence": 0.82, "raw_class": "left_hook"},
        {"timestamp": base_ts + 2.0, "punch_type": "jab", "confidence": 0.79, "raw_class": "jab"},
        {"timestamp": base_ts + 3.5, "punch_type": "right_uppercut", "confidence": 0.85, "raw_class": "right_uppercut"},
    ]


@pytest.fixture
def sample_imu_impacts():
    """Sample IMU pad impact events for fusion testing."""
    base_ts = time.time()
    return [
        {"timestamp": base_ts + 0.05, "pad": "centre", "level": "medium", "force_normalized": 0.66},
        {"timestamp": base_ts + 0.55, "pad": "centre", "level": "hard", "force_normalized": 1.0},
        {"timestamp": base_ts + 1.25, "pad": "left", "level": "light", "force_normalized": 0.33},
        {"timestamp": base_ts + 2.08, "pad": "centre", "level": "medium", "force_normalized": 0.66},
        {"timestamp": base_ts + 3.58, "pad": "right", "level": "hard", "force_normalized": 1.0},
    ]


@pytest.fixture
def sample_defense_events():
    """Sample defense evaluation data for fusion testing."""
    return {
        "arm_events": [
            {"arm": "left", "contact": False, "timestamp": 1.0},
        ],
        "cv_blocks": [
            {"confidence": 0.85, "timestamp": 1.1},
        ],
        "tracking_snapshots": [
            {"lateral_displacement": 5.0, "depth_displacement": 0.02, "timestamp": 1.05},
            {"lateral_displacement": 8.0, "depth_displacement": 0.03, "timestamp": 1.1},
        ],
    }


# ── Preset fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_preset():
    """Sample preset data."""
    return {
        "name": "Quick Jab Drill",
        "preset_type": "training",
        "config_json": json.dumps({
            "combo": "jab-cross",
            "rounds": 3,
            "work_time_sec": 120,
            "rest_time_sec": 45,
            "speed": "medium",
            "difficulty": "beginner",
        }),
        "description": "A quick jab-cross drill for warmup",
        "tags": "warmup,beginner",
    }


# ── State machine fixtures ───────────────────────────────────────────────────

@pytest.fixture
def action_labels():
    """Standard action labels used by the state machine."""
    return ["jab", "cross", "left_hook", "right_hook",
            "left_uppercut", "right_uppercut", "block", "idle"]


@pytest.fixture
def state_machine_config():
    """Default configuration for CausalActionStateMachine."""
    return {
        "enter_consecutive": 2,
        "exit_consecutive": 2,
        "min_hold_steps": 2,
        "sustain_confidence": 0.78,
        "peak_drop_threshold": 0.02,
    }
