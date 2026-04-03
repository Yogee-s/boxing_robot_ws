"""Configuration loader for BoxBunny.

Loads YAML config files and provides typed access via dataclasses.
"""
import os
import yaml
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("boxbunny.config")


@dataclass
class CVConfig:
    """CV pipeline configuration."""
    checkpoint_path: str = ""
    yolo_model_path: str = ""
    device: str = "cuda:0"
    inference_interval: int = 1
    window_size: int = 12
    min_confidence: float = 0.4
    ema_alpha: float = 0.35
    hysteresis_margin: float = 0.12
    min_hold_frames: int = 3
    block_consecutive_needed: int = 4
    state_enter_consecutive: int = 2
    state_exit_consecutive: int = 2
    state_min_hold_steps: int = 2
    state_sustain_confidence: float = 0.78
    state_peak_drop_threshold: float = 0.02


@dataclass
class FusionConfig:
    """CV + IMU fusion configuration."""
    fusion_window_ms: int = 200
    cv_unconfirmed_confidence_penalty: float = 0.3
    reclassify_min_secondary_confidence: float = 0.25
    imu_debounce_ms: int = 150
    defense_window_ms: int = 500
    slip_lateral_threshold_px: float = 40.0
    slip_depth_threshold_m: float = 0.15
    dodge_lateral_threshold_px: float = 20.0
    dodge_depth_threshold_m: float = 0.08
    block_cv_confidence_min: float = 0.3
    # Enhanced fusion: frame persistence filtering
    cv_only_min_consecutive_frames: int = 3
    cv_only_min_confidence: float = 0.6
    imu_only_default_confidence: float = 0.3


@dataclass
class IMUConfig:
    """IMU node configuration."""
    nav_debounce_ms: int = 300
    nav_global_debounce_ms: int = 200
    mode_transition_ms: int = 200
    heartbeat_interval_s: float = 1.0


@dataclass
class RobotConfig:
    """Robot arm configuration."""
    serial_port: str = "/dev/ttyACM0"
    baud_rate: int = 115200
    heartbeat_hz: float = 10.0
    punch_sequences_dir: str = "data/punch_sequences"


@dataclass
class LLMConfig:
    """LLM AI Coach configuration."""
    model_path: str = "models/llm/qwen2.5-3b-instruct-q4_k_m.gguf"
    n_gpu_layers: int = -1
    n_ctx: int = 2048
    max_tokens: int = 128
    temperature: float = 0.7
    fallback_tips_path: str = "config/fallback_tips.json"


@dataclass
class HeightConfig:
    """Height auto-adjustment configuration."""
    ideal_top_fraction: float = 0.15
    deadband_px: float = 15.0
    max_iterations: int = 3
    settle_delay_ms: int = 500
    min_depth_m: float = 0.5
    max_depth_m: float = 3.0
    no_person_timeout_s: float = 5.0


@dataclass
class TrainingConfig:
    """Default training parameters."""
    default_rounds: int = 3
    default_work_time_s: int = 180
    default_rest_time_s: int = 60
    default_speed: str = "medium"
    countdown_seconds: int = 3


@dataclass
class FreeTrainingConfig:
    """Free Training (dynamic sparring) configuration."""
    pad_counter_strikes: dict = field(default_factory=lambda: {
        "centre": ["1", "2"],
        "left": ["3", "5"],
        "right": ["4", "6"],
        "head": ["1", "2"],
    })
    counter_cooldown_ms: int = 1500
    idle_return_s: float = 5.0
    speed: str = "medium"


@dataclass
class NetworkConfig:
    """Network configuration."""
    wifi_ssid: str = "BoxBunny"
    wifi_password: str = "boxbunny2026"
    dashboard_port: int = 8080
    dashboard_host: str = "0.0.0.0"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    main_db_path: str = "data/boxbunny_main.db"
    user_data_dir: str = "data/users"
    guest_session_ttl_days: int = 7


@dataclass
class BoxBunnyConfig:
    """Root configuration."""
    cv: CVConfig = field(default_factory=CVConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    imu: IMUConfig = field(default_factory=IMUConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    height: HeightConfig = field(default_factory=HeightConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    free_training: FreeTrainingConfig = field(default_factory=FreeTrainingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)


def load_config(config_path: Optional[str] = None) -> BoxBunnyConfig:
    """Load BoxBunny configuration from YAML file.

    Falls back to defaults if file not found.
    """
    config = BoxBunnyConfig()

    if config_path is None:
        # Search common locations
        ws_root = Path(__file__).resolve().parents[3]
        candidates = [
            ws_root / "config" / "boxbunny.yaml",
            Path.home() / ".boxbunny" / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = str(candidate)
                break

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                raw = yaml.safe_load(f) or {}
            # Populate dataclasses from YAML
            for section_name, section_cls in [
                ('cv', CVConfig),
                ('fusion', FusionConfig),
                ('imu', IMUConfig),
                ('robot', RobotConfig),
                ('llm', LLMConfig),
                ('height', HeightConfig),
                ('training', TrainingConfig),
                ('free_training', FreeTrainingConfig),
                ('network', NetworkConfig),
                ('database', DatabaseConfig),
            ]:
                if section_name in raw:
                    section_data = raw[section_name]
                    section_obj = getattr(config, section_name)
                    for key, value in section_data.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.warning(
                f"Failed to load config from {config_path}: {e}. Using defaults."
            )
    else:
        logger.info("No config file found. Using defaults.")

    return config
