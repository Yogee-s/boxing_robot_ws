#!/usr/bin/env bash
# =============================================================================
# BoxBunny Setup Script
# Bootstrap the boxing robot workspace: check prerequisites, install
# dependencies, build ROS packages, download models, and initialize databases.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$WS_ROOT"

# ── Colour helpers ───────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }
header(){ echo ""; echo -e "${BOLD}${CYAN}=== $* ===${NC}"; }

# ── Status tracking ──────────────────────────────────────────────────────────

ERRORS=()
WARNINGS=()

record_error()   { ERRORS+=("$*"); fail "$*"; }
record_warning() { WARNINGS+=("$*"); warn "$*"; }

echo "========================================="
echo "  BoxBunny Setup Script v2.0"
echo "  Workspace: $WS_ROOT"
echo "========================================="

# =============================================================================
# 1. Check Prerequisites
# =============================================================================
header "1/7  Checking Prerequisites"

# Python 3.10+
if command -v python3 &>/dev/null; then
    PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 10 ]; then
        ok "Python $PY_VERSION"
    else
        record_error "Python 3.10+ required, found $PY_VERSION"
    fi
else
    record_error "Python 3 not found"
fi

# pip
if python3 -m pip --version &>/dev/null; then
    ok "pip available"
else
    record_error "pip not available (python3 -m pip failed)"
fi

# ROS 2
ROS_AVAILABLE=false
if [ -n "${ROS_DISTRO:-}" ]; then
    ok "ROS 2 ($ROS_DISTRO) sourced"
    ROS_AVAILABLE=true
elif [ -f /opt/ros/humble/setup.bash ]; then
    info "Sourcing ROS 2 Humble..."
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    ok "ROS 2 Humble sourced"
    ROS_AVAILABLE=true
elif [ -f /opt/ros/iron/setup.bash ]; then
    info "Sourcing ROS 2 Iron..."
    # shellcheck disable=SC1091
    source /opt/ros/iron/setup.bash
    ok "ROS 2 Iron sourced"
    ROS_AVAILABLE=true
else
    record_warning "ROS 2 not found -- colcon build will be skipped"
fi

# colcon
COLCON_AVAILABLE=false
if command -v colcon &>/dev/null; then
    ok "colcon available"
    COLCON_AVAILABLE=true
else
    if $ROS_AVAILABLE; then
        record_warning "colcon not found -- install with: pip install colcon-common-extensions"
    fi
fi

# CUDA
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "unknown")
    ok "CUDA $CUDA_VERSION"
elif [ -d /usr/local/cuda ]; then
    ok "CUDA directory found at /usr/local/cuda (nvcc not in PATH)"
else
    record_warning "CUDA not detected -- GPU acceleration unavailable"
fi

# PyTorch + CUDA
if python3 -c "import torch; print(f'  PyTorch {torch.__version__}')" 2>/dev/null; then
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        ok "PyTorch CUDA available"
    else
        warn "PyTorch installed but CUDA not available (CPU fallback)"
    fi
else
    info "PyTorch not yet installed (will install from requirements.txt)"
fi

# SQLite3
if python3 -c "import sqlite3" 2>/dev/null; then
    ok "SQLite3 Python module available"
else
    record_error "Python sqlite3 module not available"
fi

# =============================================================================
# 2. Install Python Dependencies
# =============================================================================
header "2/7  Installing Python Dependencies"

if [ -f "$WS_ROOT/requirements.txt" ]; then
    info "Installing from requirements.txt ..."
    if python3 -m pip install --user -r "$WS_ROOT/requirements.txt" 2>&1 | tail -3; then
        ok "Core dependencies installed"
    else
        record_error "pip install -r requirements.txt failed"
    fi
else
    record_error "requirements.txt not found"
fi

# Jetson-specific dependencies
if [ -f "$WS_ROOT/requirements-jetson.txt" ]; then
    info "Installing Jetson-specific dependencies..."
    if python3 -m pip install --user -r "$WS_ROOT/requirements-jetson.txt" 2>&1 | tail -3; then
        ok "Jetson dependencies installed"
    else
        record_warning "Jetson dependencies failed (may not be on Jetson)"
    fi
fi

# Dev dependencies
if [ -f "$WS_ROOT/requirements-dev.txt" ]; then
    info "Installing dev/test dependencies..."
    if python3 -m pip install --user -r "$WS_ROOT/requirements-dev.txt" 2>&1 | tail -3; then
        ok "Dev dependencies installed"
    else
        record_warning "Dev dependencies failed (non-critical)"
    fi
fi

# =============================================================================
# 3. Build ROS 2 Workspace
# =============================================================================
header "3/7  Building ROS 2 Workspace"

if $ROS_AVAILABLE && $COLCON_AVAILABLE; then
    info "Running colcon build --symlink-install ..."
    if colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5; then
        ok "colcon build complete"
        if [ -f "$WS_ROOT/install/setup.bash" ]; then
            # shellcheck disable=SC1091
            source "$WS_ROOT/install/setup.bash"
            ok "Workspace overlay sourced"
        fi
    else
        record_error "colcon build failed"
    fi
else
    record_warning "Skipping colcon build (ROS 2 or colcon not available)"
fi

# =============================================================================
# 4. Download Models
# =============================================================================
header "4/7  Downloading Models"

if [ -f "$SCRIPT_DIR/download_models.sh" ]; then
    info "Running model download script..."
    if bash "$SCRIPT_DIR/download_models.sh"; then
        ok "Model download script completed"
    else
        record_warning "Model download had issues"
    fi
else
    record_warning "download_models.sh not found"
fi

# =============================================================================
# 5. Initialize Databases
# =============================================================================
header "5/7  Initializing Databases"

SCHEMA_DIR="$WS_ROOT/data/schema"

if [ -f "$SCHEMA_DIR/main_schema.sql" ]; then
    info "Initializing main database..."
    if command -v sqlite3 &>/dev/null; then
        sqlite3 "$WS_ROOT/data/boxbunny_main.db" < "$SCHEMA_DIR/main_schema.sql" 2>/dev/null || true
        ok "Main database initialized"
    else
        # Fallback to Python
        python3 -c "
import sqlite3, pathlib
schema = pathlib.Path('$SCHEMA_DIR/main_schema.sql').read_text()
conn = sqlite3.connect('$WS_ROOT/data/boxbunny_main.db')
conn.executescript(schema)
conn.close()
print('  Main database initialized via Python')
" 2>/dev/null && ok "Main database initialized" || record_warning "Main database init failed"
    fi
else
    record_warning "Main schema not found at $SCHEMA_DIR/main_schema.sql"
fi

# Create demo user via the DatabaseManager
info "Creating demo user..."
python3 -c "
import sys
sys.path.insert(0, '$WS_ROOT/src/boxbunny_dashboard')
try:
    from boxbunny_dashboard.db.manager import DatabaseManager
    db = DatabaseManager('$WS_ROOT/data')
    result = db.create_user('demo', 'demo123', 'Demo User', 'individual', 'beginner')
    if result:
        print('  Demo user created (username: demo, password: demo123)')
    else:
        print('  Demo user already exists')
except Exception as e:
    print(f'  Demo user creation skipped: {e}')
" 2>/dev/null && ok "Database users ready" || record_warning "Demo user creation failed (non-critical)"

# =============================================================================
# 6. Build Knowledge Base Manifest
# =============================================================================
header "6/7  Verifying Knowledge Base"

KB_DIR="$WS_ROOT/data/boxing_knowledge"
MANIFEST="$KB_DIR/manifest.json"

if [ -f "$MANIFEST" ]; then
    DOC_COUNT=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['total_documents'])" 2>/dev/null || echo "?")
    ok "Knowledge base manifest found ($DOC_COUNT documents)"

    # Verify all listed files exist
    MISSING=0
    while IFS= read -r doc_path; do
        if [ ! -f "$KB_DIR/$doc_path" ]; then
            record_warning "Missing knowledge base document: $doc_path"
            MISSING=$((MISSING + 1))
        fi
    done < <(python3 -c "
import json
manifest = json.load(open('$MANIFEST'))
for doc in manifest['documents']:
    print(doc['path'])
" 2>/dev/null || true)

    if [ "$MISSING" -eq 0 ]; then
        ok "All knowledge base documents present"
    fi
else
    record_warning "Knowledge base manifest not found at $MANIFEST"
fi

# =============================================================================
# 7. Verify Components and Print Status Report
# =============================================================================
header "7/7  Component Verification"

# Required directories
DIRS=(
    "$WS_ROOT/data/users"
    "$WS_ROOT/data/punch_sequences"
    "$WS_ROOT/data/reaction_drill"
    "$WS_ROOT/data/shadow_sparring"
    "$WS_ROOT/models/llm"
    "$WS_ROOT/models/mediapipe"
    "$WS_ROOT/logs"
)
for dir in "${DIRS[@]}"; do
    mkdir -p "$dir"
done
ok "Required directories present"

# CV model
if [ -f "$WS_ROOT/action_prediction/model/best_model.pth" ]; then
    ok "CV action prediction model: present"
else
    record_warning "CV model missing (place best_model.pth in action_prediction/model/)"
fi

# YOLO model
if [ -f "$WS_ROOT/action_prediction/model/yolo26n-pose.pt" ]; then
    ok "YOLO pose model: present"
else
    record_warning "YOLO pose model missing"
fi

# LLM model
if ls "$WS_ROOT/models/llm/"*.gguf 1>/dev/null 2>&1; then
    LLM_SIZE=$(du -h "$WS_ROOT/models/llm/"*.gguf 2>/dev/null | head -1 | cut -f1)
    ok "LLM model: present ($LLM_SIZE)"
else
    record_warning "LLM model missing (run: scripts/download_models.sh)"
fi

# Punch sequences
SEQ_COUNT=$(ls "$WS_ROOT/data/punch_sequences/"*.json 2>/dev/null | wc -l || echo "0")
ok "Punch sequences: $SEQ_COUNT files"

# ROS messages
if $ROS_AVAILABLE; then
    if ros2 interface show boxbunny_msgs/msg/ConfirmedPunch 2>/dev/null | head -1 | grep -q "float64"; then
        ok "ROS messages: built and available"
    else
        record_warning "ROS messages not built -- run colcon build"
    fi
fi

# =============================================================================
# Final Summary
# =============================================================================
echo ""
echo "========================================="
echo "  Setup Summary"
echo "========================================="
echo ""
echo "  Workspace:  $WS_ROOT"
echo "  Python:     $(python3 --version 2>&1)"
echo "  ROS 2:      ${ROS_DISTRO:-not detected}"
echo ""

if [ ${#ERRORS[@]} -eq 0 ] && [ ${#WARNINGS[@]} -eq 0 ]; then
    echo -e "  ${GREEN}All checks passed. BoxBunny is ready.${NC}"
elif [ ${#ERRORS[@]} -eq 0 ]; then
    echo -e "  ${YELLOW}Completed with ${#WARNINGS[@]} warning(s):${NC}"
    for w in "${WARNINGS[@]}"; do
        echo -e "    ${YELLOW}- $w${NC}"
    done
else
    echo -e "  ${RED}Completed with ${#ERRORS[@]} error(s) and ${#WARNINGS[@]} warning(s):${NC}"
    for e in "${ERRORS[@]}"; do
        echo -e "    ${RED}[ERROR] $e${NC}"
    done
    for w in "${WARNINGS[@]}"; do
        echo -e "    ${YELLOW}[WARN]  $w${NC}"
    done
fi

echo ""
echo "  Quick Start:"
echo "    source install/setup.bash"
echo "    ros2 launch boxbunny_core boxbunny_dev.launch.py"
echo "    Dashboard: http://localhost:8080"
echo ""
echo "========================================="
