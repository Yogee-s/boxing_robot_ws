float64 timestamp
string punch_type        # classified punch type
string pad               # which pad was hit
string level             # "light", "medium", "hard" (empty if IMU unavailable)
float32 force_normalized # 0.33/0.66/1.0 (0.0 if IMU unavailable)
float32 cv_confidence    # CV model confidence
bool imu_confirmed       # true if pad IMU impact matched
bool cv_confirmed        # true if CV detection matched
