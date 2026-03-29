#!/bin/bash
# Test the dashboard API endpoints
set +e
cd /home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws
source /opt/ros/humble/setup.bash && source install/setup.bash

python3 -m boxbunny_dashboard.server &
sleep 3

echo "=== Testing API ==="
echo ""
echo "--- Health Check ---"
curl -s http://localhost:8080/api/health \
    | python3 -m json.tool 2>/dev/null \
    || echo "(server not responding)"

echo ""
echo "--- Login as Alex ---"
TOKEN=$(curl -s -X POST http://localhost:8080/api/auth/login \
    -H 'Content-Type: application/json' \
    -d '{"username": "alex", "password": "boxing123"}' \
    | python3 -c \
    "import sys,json; print(json.load(sys.stdin).get('token',''))" \
    2>/dev/null)
echo "Token: ${TOKEN:0:20}..."

if [ -n "$TOKEN" ]; then
    echo ""
    echo "--- Alex's Session History ---"
    curl -s http://localhost:8080/api/sessions/history \
        -H "Authorization: Bearer $TOKEN" \
        | python3 -m json.tool 2>/dev/null | head -30

    echo ""
    echo "--- Alex's Gamification Profile ---"
    curl -s http://localhost:8080/api/gamification/profile \
        -H "Authorization: Bearer $TOKEN" \
        | python3 -m json.tool 2>/dev/null
fi

pkill -f 'boxbunny_dashboard.server' 2>/dev/null
echo ""
echo "Dashboard test complete"
