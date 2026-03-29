"""Central constants for BoxBunny boxing training robot.

All ROS topic names, message constants, and system-wide enumerations live here.
Modify topic names here to change them across the entire system.
"""


class Topics:
    """ROS 2 topic names -- single source of truth."""

    # ── IMU (from Teensy) ────────────────────────────────────────────────
    IMU_PAD_IMPACT = "/boxbunny/imu/pad/impact"
    IMU_ARM_STRIKE = "/boxbunny/imu/arm/strike"
    IMU_STATUS = "/boxbunny/imu/status"

    # ── IMU (processed by imu_node) ──────────────────────────────────────
    IMU_PUNCH_EVENT = "/boxbunny/imu/punch_event"
    IMU_NAV_EVENT = "/boxbunny/imu/nav_event"
    IMU_ARM_EVENT = "/boxbunny/imu/arm_event"

    # ── CV (from cv_node) ────────────────────────────────────────────────
    CV_DETECTION = "/boxbunny/cv/detection"
    CV_POSE = "/boxbunny/cv/pose"
    CV_USER_TRACKING = "/boxbunny/cv/user_tracking"
    CV_STATUS = "/boxbunny/cv/status"

    # ── Fused (from punch_processor) ─────────────────────────────────────
    PUNCH_CONFIRMED = "/boxbunny/punch/confirmed"
    PUNCH_DEFENSE = "/boxbunny/punch/defense"
    PUNCH_SESSION_SUMMARY = "/boxbunny/punch/session_summary"

    # ── Robot arm ────────────────────────────────────────────────────────
    ROBOT_COMMAND = "/boxbunny/robot/command"
    ROBOT_HEIGHT = "/boxbunny/robot/height"
    ROBOT_ROUND_CONTROL = "/boxbunny/robot/round_control"
    ROBOT_STATUS = "/boxbunny/robot/status"

    # ── Session ──────────────────────────────────────────────────────────
    SESSION_STATE = "/boxbunny/session/state"
    SESSION_CONFIG = "/boxbunny/session/config"

    # ── Drills ───────────────────────────────────────────────────────────
    DRILL_DEFINITION = "/boxbunny/drill/definition"
    DRILL_EVENT = "/boxbunny/drill/event"
    DRILL_PROGRESS = "/boxbunny/drill/progress"

    # ── AI Coach ─────────────────────────────────────────────────────────
    COACH_TIP = "/boxbunny/coach/tip"

    # ── Camera (RealSense) ───────────────────────────────────────────────
    CAMERA_COLOR = "/camera/color/image_raw"
    CAMERA_DEPTH = "/camera/aligned_depth_to_color/image_raw"


class Services:
    """ROS 2 service names."""

    START_SESSION = "/boxbunny/session/start"
    END_SESSION = "/boxbunny/session/end"
    START_DRILL = "/boxbunny/drill/start"
    SET_IMU_MODE = "/boxbunny/imu/set_mode"
    CALIBRATE_IMU = "/boxbunny/imu/calibrate"
    GENERATE_LLM = "/boxbunny/llm/generate"


class PunchType:
    """Punch type identifiers matching CV model output."""

    JAB = "jab"
    CROSS = "cross"
    LEFT_HOOK = "left_hook"
    RIGHT_HOOK = "right_hook"
    LEFT_UPPERCUT = "left_uppercut"
    RIGHT_UPPERCUT = "right_uppercut"
    BLOCK = "block"
    IDLE = "idle"

    ALL_ACTIONS = [
        JAB, CROSS, LEFT_HOOK, RIGHT_HOOK,
        LEFT_UPPERCUT, RIGHT_UPPERCUT, BLOCK, IDLE,
    ]
    OFFENSIVE = [
        JAB, CROSS, LEFT_HOOK, RIGHT_HOOK,
        LEFT_UPPERCUT, RIGHT_UPPERCUT,
    ]

    # Punch codes for robot commands (1-indexed)
    CODE_MAP = {
        "1": JAB,
        "2": CROSS,
        "3": LEFT_HOOK,
        "4": RIGHT_HOOK,
        "5": LEFT_UPPERCUT,
        "6": RIGHT_UPPERCUT,
    }


class PadLocation:
    """Pad IMU locations."""

    LEFT = "left"
    CENTRE = "centre"
    RIGHT = "right"
    HEAD = "head"
    ALL = [LEFT, CENTRE, RIGHT, HEAD]

    # Valid punch types per pad (for fusion constraints)
    VALID_PUNCHES = {
        LEFT: [
            PunchType.JAB, PunchType.CROSS,
            PunchType.LEFT_HOOK, PunchType.LEFT_UPPERCUT,
        ],
        CENTRE: [
            PunchType.JAB, PunchType.CROSS,
            PunchType.LEFT_UPPERCUT, PunchType.RIGHT_UPPERCUT,
        ],
        RIGHT: [
            PunchType.JAB, PunchType.CROSS,
            PunchType.RIGHT_HOOK, PunchType.RIGHT_UPPERCUT,
        ],
        HEAD: PunchType.OFFENSIVE,  # all offensive punches valid on head pad
    }


class ImpactLevel:
    """Impact force levels from Teensy classification."""

    LIGHT = "light"
    MEDIUM = "medium"
    HARD = "hard"
    ALL = [LIGHT, MEDIUM, HARD]
    FORCE_MAP = {LIGHT: 0.33, MEDIUM: 0.66, HARD: 1.0}


class ArmSide:
    """Robot arm sides."""

    LEFT = "left"
    RIGHT = "right"


class SessionState:
    """Session state values."""

    IDLE = "idle"
    COUNTDOWN = "countdown"
    ACTIVE = "active"
    REST = "rest"
    COMPLETE = "complete"


class TrainingMode:
    """Training mode identifiers."""

    TRAINING = "training"
    SPARRING = "sparring"
    FREE = "free"
    POWER = "power"
    STAMINA = "stamina"
    REACTION = "reaction"
    ALL = [TRAINING, SPARRING, FREE, POWER, STAMINA, REACTION]


class Difficulty:
    """Difficulty levels."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    ALL = [BEGINNER, INTERMEDIATE, ADVANCED]


class Speed:
    """Robot speed settings."""

    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"


class DefenseType:
    """Defense event types."""

    BLOCK = "block"
    SLIP = "slip"
    DODGE = "dodge"
    HIT = "hit"
    UNKNOWN = "unknown"


class NavCommand:
    """IMU navigation commands."""

    PREV = "prev"
    NEXT = "next"
    ENTER = "enter"
    BACK = "back"

    # Pad-to-command mapping
    PAD_MAP = {
        PadLocation.LEFT: PREV,
        PadLocation.RIGHT: NEXT,
        PadLocation.CENTRE: ENTER,
        PadLocation.HEAD: BACK,
    }
