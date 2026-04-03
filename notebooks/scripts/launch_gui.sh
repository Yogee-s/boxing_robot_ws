#!/usr/bin/env bash
# Launch the main BoxBunny touchscreen GUI for visual inspection.
# Close the window to end.
set +e
cd /home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws
source /opt/ros/humble/setup.bash && source install/setup.bash

export QT_QPA_PLATFORM=xcb
export QT_QPA_PLATFORM_PLUGIN_PATH=$(python3 -c "import PySide6; print(PySide6.__path__[0])")/Qt/plugins/platforms
unset QT_PLUGIN_PATH

cleanup() {
    kill -- -$$ 2>/dev/null
    sleep 0.5
    kill -9 -- -$$ 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "Launching BoxBunny GUI..."
echo "Close the window to end the test."
echo ""

python3 -c "
import sys, os, signal, atexit
sys.path.insert(0, 'src/boxbunny_gui')
os.environ.pop('QT_PLUGIN_PATH', None)
from boxbunny_gui.app import BoxBunnyApp
app = BoxBunnyApp()
def _shutdown(*a):
    try: app._shutdown()
    except: pass
    try: app._window.close()
    except: pass
    os._exit(0)
signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)
atexit.register(_shutdown)
app.run()
" 2>&1 || true
