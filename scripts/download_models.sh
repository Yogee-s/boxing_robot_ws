#!/usr/bin/env bash
# =============================================================================
# BoxBunny Model Download Script
# Downloads required model files. Idempotent -- skips files that already exist.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$WS_ROOT/models"

# ── Colour helpers ───────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }

echo "=== BoxBunny Model Download ==="
echo ""

# ── Configuration ────────────────────────────────────────────────────────────

LLM_DIR="$MODELS_DIR/llm"
LLM_FILENAME="qwen2.5-3b-instruct-q4_k_m.gguf"
LLM_FILE="$LLM_DIR/$LLM_FILENAME"
LLM_HF_REPO="Qwen/Qwen2.5-3B-Instruct-GGUF"
LLM_URL="https://huggingface.co/$LLM_HF_REPO/resolve/main/$LLM_FILENAME"
LLM_MIN_SIZE_MB=1800  # Minimum expected size to consider download complete

# =============================================================================
# Pre-flight Checks
# =============================================================================

# Determine download tool
DOWNLOADER=""
if command -v huggingface-cli &>/dev/null; then
    DOWNLOADER="hf-cli"
    info "Download tool: huggingface-cli"
elif command -v wget &>/dev/null; then
    DOWNLOADER="wget"
    info "Download tool: wget"
elif command -v curl &>/dev/null; then
    DOWNLOADER="curl"
    info "Download tool: curl"
else
    fail "No download tool found. Install wget, curl, or huggingface-cli."
    exit 1
fi

# Check disk space (need at least 3GB free)
AVAILABLE_MB=$(df -BM "$WS_ROOT" 2>/dev/null | tail -1 | awk '{gsub(/M/,"",$4); print $4}' || echo "99999")
if [ "${AVAILABLE_MB:-99999}" -lt 3000 ] 2>/dev/null; then
    warn "Low disk space: ${AVAILABLE_MB}MB available. LLM model requires ~2GB."
    echo -n "  Continue anyway? [y/N] "
    read -r response
    if [[ ! "${response:-}" =~ ^[yY]$ ]]; then
        info "Aborting download."
        exit 0
    fi
fi

# =============================================================================
# Helper Functions
# =============================================================================

download_file() {
    local url="$1"
    local dest="$2"

    case "$DOWNLOADER" in
        hf-cli)
            # huggingface-cli downloads to a directory
            local dest_dir
            dest_dir="$(dirname "$dest")"
            local dest_name
            dest_name="$(basename "$dest")"
            huggingface-cli download "$LLM_HF_REPO" \
                "$dest_name" \
                --local-dir "$dest_dir" \
                --local-dir-use-symlinks False
            ;;
        wget)
            wget -q --show-progress -O "$dest" "$url"
            ;;
        curl)
            curl -L --progress-bar -o "$dest" "$url"
            ;;
    esac
}

file_size_mb() {
    local filepath="$1"
    if [ ! -f "$filepath" ]; then
        echo 0
        return
    fi
    local size_bytes
    size_bytes=$(stat -c%s "$filepath" 2>/dev/null || stat -f%z "$filepath" 2>/dev/null || echo 0)
    echo $((size_bytes / 1048576))
}

# =============================================================================
# 1. LLM Model: Qwen2.5-3B-Instruct Q4_K_M GGUF
# =============================================================================

echo ""
info "--- LLM Model: Qwen2.5-3B-Instruct Q4_K_M ---"

mkdir -p "$LLM_DIR"

if [ -f "$LLM_FILE" ]; then
    SIZE_MB=$(file_size_mb "$LLM_FILE")
    if [ "$SIZE_MB" -ge "$LLM_MIN_SIZE_MB" ]; then
        ok "Already downloaded: $LLM_FILENAME (${SIZE_MB}MB)"
    else
        warn "File exists but appears incomplete (${SIZE_MB}MB < ${LLM_MIN_SIZE_MB}MB). Re-downloading..."
        rm -f "$LLM_FILE"

        info "Downloading $LLM_FILENAME (~2GB) ..."
        if download_file "$LLM_URL" "$LLM_FILE"; then
            SIZE_MB=$(file_size_mb "$LLM_FILE")
            if [ "$SIZE_MB" -ge "$LLM_MIN_SIZE_MB" ]; then
                ok "Download complete: ${SIZE_MB}MB"
            else
                fail "Downloaded file too small (${SIZE_MB}MB). Download may have failed."
                rm -f "$LLM_FILE"
            fi
        else
            fail "Download failed"
            rm -f "$LLM_FILE"
        fi
    fi
else
    info "Downloading $LLM_FILENAME (~2GB) ..."
    info "Source: $LLM_URL"
    info "Destination: $LLM_FILE"
    echo ""

    if download_file "$LLM_URL" "$LLM_FILE"; then
        SIZE_MB=$(file_size_mb "$LLM_FILE")
        if [ "$SIZE_MB" -ge "$LLM_MIN_SIZE_MB" ]; then
            ok "Download complete: ${SIZE_MB}MB"
        else
            fail "Downloaded file too small (${SIZE_MB}MB). Download may have failed."
            rm -f "$LLM_FILE"
        fi
    else
        fail "Download failed"
        rm -f "$LLM_FILE"
        echo ""
        info "Manual download:"
        info "  URL:  $LLM_URL"
        info "  Save: $LLM_FILE"
    fi
fi

# =============================================================================
# 2. CV Model Check (cannot auto-download -- custom trained)
# =============================================================================

echo ""
info "--- CV Model Check ---"

CV_MODEL="$WS_ROOT/action_prediction/model/best_model.pth"
if [ -f "$CV_MODEL" ]; then
    SIZE_MB=$(file_size_mb "$CV_MODEL")
    ok "CV model present: $(basename "$CV_MODEL") (${SIZE_MB}MB)"
else
    warn "CV model not found at: $CV_MODEL"
    info "This is a custom-trained model and cannot be auto-downloaded."
fi

# =============================================================================
# 3. YOLO Pose Model Check
# =============================================================================

echo ""
info "--- YOLO Pose Model Check ---"

YOLO_MODEL="$WS_ROOT/action_prediction/model/yolo26n-pose.pt"
if [ -f "$YOLO_MODEL" ]; then
    SIZE_MB=$(file_size_mb "$YOLO_MODEL")
    ok "YOLO pose model present: $(basename "$YOLO_MODEL") (${SIZE_MB}MB)"
else
    warn "YOLO pose model not found at: $YOLO_MODEL"
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=== Model Status Summary ==="
echo ""

if [ -f "$LLM_FILE" ]; then
    echo -e "  LLM (Qwen2.5-3B Q4_K_M):  ${GREEN}Present${NC} ($(file_size_mb "$LLM_FILE")MB)"
else
    echo -e "  LLM (Qwen2.5-3B Q4_K_M):  ${RED}Missing${NC}"
fi

if [ -f "$CV_MODEL" ]; then
    echo -e "  CV Action Prediction:      ${GREEN}Present${NC}"
else
    echo -e "  CV Action Prediction:      ${YELLOW}Missing (manual install)${NC}"
fi

if [ -f "$YOLO_MODEL" ]; then
    echo -e "  YOLO Pose:                 ${GREEN}Present${NC}"
else
    echo -e "  YOLO Pose:                 ${YELLOW}Missing${NC}"
fi

echo ""
echo "=== Model Download Complete ==="
