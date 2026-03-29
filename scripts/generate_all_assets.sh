#!/bin/bash
# Generate all BoxBunny visual and audio assets.
# Run from the workspace root:
#   bash scripts/generate_all_assets.sh
set -e

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
echo "Workspace: $WORKSPACE"
echo ""

echo "=== Step 1: Generate Sound Effects ==="
python3 "$WORKSPACE/scripts/generate_sounds.py"
echo ""

echo "=== Step 2: Generate PWA PNG Icons ==="
python3 "$WORKSPACE/scripts/generate_icons.py"
echo ""

echo "=== Step 3: Font Check ==="
FONTS_DIR="$WORKSPACE/src/boxbunny_gui/assets/fonts"
if ls "$FONTS_DIR"/Inter*.ttf 1>/dev/null 2>&1; then
    echo "Inter font files found:"
    ls -la "$FONTS_DIR"/Inter*.ttf
else
    echo "WARNING: Inter font files not found in $FONTS_DIR"
    echo "See FONT_README.txt for manual installation instructions."
fi
echo ""

echo "=== Asset Summary ==="
echo "  Sounds:       $(find "$WORKSPACE/src/boxbunny_gui/assets/sounds" -name "*.wav" 2>/dev/null | wc -l) WAV files"
echo "  GUI Icons:    $(find "$WORKSPACE/src/boxbunny_gui/assets/icons" -name "*.svg" 2>/dev/null | wc -l) SVG files"
echo "  Fonts:        $(find "$WORKSPACE/src/boxbunny_gui/assets/fonts" -name "*.ttf" 2>/dev/null | wc -l) TTF files"
echo "  PWA Icons:    $(find "$WORKSPACE/src/boxbunny_dashboard/frontend/public" -maxdepth 1 \( -name "*.png" -o -name "*.svg" \) 2>/dev/null | wc -l) files"
echo "  Rank Badges:  $(find "$WORKSPACE/src/boxbunny_dashboard/frontend/public/ranks" -name "*.svg" 2>/dev/null | wc -l) SVG files"
echo "  Achievements: $(find "$WORKSPACE/src/boxbunny_dashboard/frontend/public/achievements" -name "*.svg" 2>/dev/null | wc -l) SVG files"
echo ""

TOTAL=$(find \
    "$WORKSPACE/src/boxbunny_gui/assets" \
    "$WORKSPACE/src/boxbunny_dashboard/frontend/public" \
    -type f ! -name "manifest.json" 2>/dev/null | wc -l)
echo "Total asset files: $TOTAL"
echo ""
echo "Done!"
