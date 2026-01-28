std_msgs/Header header
string action_label          # Current predicted action (jab, cross, block, etc.)
float32[] probabilities      # Probability per class
string[] class_labels        # Class label names
float32 confidence           # Top prediction confidence
