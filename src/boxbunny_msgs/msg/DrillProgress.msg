std_msgs/Header header
string drill_name
int32 current_step           # Current step in sequence (0-indexed)
int32 total_steps
string[] expected_actions
string[] detected_actions
bool[] step_completed
float32 elapsed_time_s
string status                # "in_progress", "success", "failed", "timeout"
