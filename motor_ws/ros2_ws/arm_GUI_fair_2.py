#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
import threading
import time
import json
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# --- SHARED ROS 2 NODE ---
class RobotNode(Node):
    def __init__(self):
        super().__init__('combined_gui_node')
        
        # Publisher & Subscriber
        self.pub_cmd = self.create_publisher(Float64MultiArray, 'motor_commands', 10)
        self.sub_feedback = self.create_subscription(
            Float64MultiArray, 'motor_feedback', self.feedback_callback, 10
        )
        
        # Trigger Subscriber
        self.sub_trigger = None
        self.latest_trigger_msg = None
        self.setup_trigger_subscription('/robot/robot_action_trigger')
        
        # State Vectors [M1, M2, M3, M4]
        self.actual_pos = [0.0] * 4
        self.target_pos = [0.0] * 4
        self.target_speed = [5.0] * 4
        self.motor_enabled = False
        
        self.last_msg_time = 0
        self.rx_count = 0
        
        self.running = True
        self.thread = threading.Thread(target=self.heartbeat_loop, daemon=True)
        self.thread.start()

    def setup_trigger_subscription(self, topic_name):
        if self.sub_trigger:
            self.destroy_subscription(self.sub_trigger)
        
        self.sub_trigger = self.create_subscription(
            String, topic_name, self.trigger_callback, 10
        )
        self.get_logger().info(f"Listening for triggers on: {topic_name}")

    def trigger_callback(self, msg):
        self.latest_trigger_msg = msg.data

    def feedback_callback(self, msg):
        self.last_msg_time = time.time()
        # Parse [P1, P2, P3, P4, Count]
        if len(msg.data) >= 4:
            self.actual_pos = list(msg.data[0:4])
            if len(msg.data) >= 5:
                self.rx_count = int(msg.data[4])

    def heartbeat_loop(self):
        """Sends commands at 10Hz to satisfy motor watchdog"""
        while self.running and rclpy.ok():
            msg = Float64MultiArray()
            # Protocol: [P1, P2, P3, P4, S1, S2, S3, S4, Mode]
            payload = []
            payload.extend(self.target_pos)
            payload.extend(self.target_speed)
            payload.append(1.0 if self.motor_enabled else 0.0)
            msg.data = payload
            
            self.pub_cmd.publish(msg)
            time.sleep(0.1)

    def set_target_arm(self, arm_idx, p1, p2, speed=None):
        # arm_idx: 0=Left (M1,M2), 1=Right (M3,M4)
        offset = arm_idx * 2
        self.target_pos[offset] = p1
        self.target_pos[offset+1] = p2
        if speed:
            self.target_speed[offset] = speed
            self.target_speed[offset+1] = speed

    def set_targets_all(self, targets, speeds=None):
        if len(targets) == 4:
            self.target_pos = targets
        
        if speeds:
            if isinstance(speeds, list) and len(speeds) == 4:
                self.target_speed = speeds
            elif isinstance(speeds, (float, int)):
                self.target_speed = [float(speeds)] * 4

    def shutdown(self):
        self.running = False
        self.thread.join()

# --- UTILS ---
def parse_point(step, current_actuals):
    raw_pos = step.get('pos', [])
    if len(raw_pos) == 2: # Legacy 2-motor file
        return [raw_pos[0], raw_pos[1], current_actuals[2], current_actuals[3]]
    elif len(raw_pos) >= 4:
        return raw_pos[0:4]
    return [0.0]*4

def parse_speed(step):
    s1 = step.get('spd_l', 5.0)
    s2 = step.get('spd_r', 5.0)
    return [s1, s1, s2, s2]

def wait_for_arrival(node, targets, active_flag_check=None):
    timeout = 8.0 
    start_t = time.time()
    while time.time() - start_t < timeout:
        # Allow external interruption
        if active_flag_check and not active_flag_check():
            return False

        all_good = True
        for i in range(4):
            if abs(node.actual_pos[i] - targets[i]) > 0.2:
                all_good = False
                break
        if all_good: return True
        time.sleep(0.05)
    return False

# --- TAB 1: MANUAL TEACHING & SEQUENCER ---
class ManualTab:
    def __init__(self, parent, node):
        self.parent = parent
        self.node = node
        self.sequence = []
        self.setup_ui()

    def setup_ui(self):
        # Main Split
        main_pane = ttk.PanedWindow(self.parent, orient="horizontal")
        main_pane.pack(fill="both", expand=True, padx=5, pady=5)

        # Left Arm Frame
        self.frame_left = ttk.LabelFrame(main_pane, text="Left Group (M1, M2)", padding=5)
        main_pane.add(self.frame_left, weight=1)
        self.setup_arm_panel(self.frame_left, "Left", 0)

        # Right Arm Frame
        self.frame_right = ttk.LabelFrame(main_pane, text="Right Group (M3, M4)", padding=5)
        main_pane.add(self.frame_right, weight=1)
        self.setup_arm_panel(self.frame_right, "Right", 1)

        # Sequencer Controls (Bottom)
        seq_frame = ttk.LabelFrame(self.parent, text="Sequencer & Recording", padding=5)
        seq_frame.pack(fill="x", padx=5, pady=5)
        
        # Tools
        tool_frame = ttk.Frame(seq_frame)
        tool_frame.pack(fill="x", pady=2)
        ttk.Button(tool_frame, text="Record Snapshot", command=self.record_point).pack(side="left", padx=2)
        ttk.Button(tool_frame, text="Delete Selected", command=self.delete_point).pack(side="left", padx=2)
        ttk.Button(tool_frame, text="Clear All", command=self.clear_sequence).pack(side="left", padx=2)
        
        # IO
        io_frame = ttk.Frame(seq_frame)
        io_frame.pack(fill="x", pady=2)
        ttk.Button(io_frame, text="Save to File", command=self.save_sequence).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Button(io_frame, text="Load from File", command=self.load_sequence).pack(side="left", fill="x", expand=True, padx=2)

        # Play
        self.btn_play = ttk.Button(seq_frame, text="▶ PLAY SEQUENCE", command=self.play_sequence)
        self.btn_play.pack(fill="x", pady=5)
        
        # Listbox
        self.listbox = tk.Listbox(seq_frame, height=8)
        self.listbox.pack(fill="both", expand=True)

    def setup_arm_panel(self, parent, name, idx):
        # Readout
        lbl_pos = ttk.Label(parent, text=f"Actual: 0.00  0.00", font=("Courier", 12))
        lbl_pos.pack(pady=2)
        setattr(self, f"lbl_pos_{name.lower()}", lbl_pos)

        # Sliders
        id_a = idx*2 + 1
        id_b = idx*2 + 2
        
        s1 = tk.Scale(parent, from_=-12.5, to=12.5, orient="horizontal", label=f"Motor {id_a}", resolution=0.1)
        s1.pack(fill="x")
        setattr(self, f"scale_{name.lower()}_1", s1)
        
        s2 = tk.Scale(parent, from_=-12.5, to=12.5, orient="horizontal", label=f"Motor {id_b}", resolution=0.1)
        s2.pack(fill="x")
        setattr(self, f"scale_{name.lower()}_2", s2)
        
        spd = tk.Scale(parent, from_=0.5, to=30.0, orient="horizontal", label="Speed (rad/s)", resolution=0.5)
        spd.set(5.0)
        spd.pack(fill="x")
        setattr(self, f"scale_{name.lower()}_spd", spd)
        
        ttk.Button(parent, text=f"Move {name} Arm", command=lambda: self.send_arm(idx, name)).pack(pady=5)

    def send_arm(self, idx, name):
        s1 = getattr(self, f"scale_{name.lower()}_1").get()
        s2 = getattr(self, f"scale_{name.lower()}_2").get()
        spd = getattr(self, f"scale_{name.lower()}_spd").get()
        self.node.set_target_arm(idx, s1, s2, spd)

    def record_point(self):
        # Records [Pos1, Pos2, Pos3, Pos4] and current Speed settings from sliders
        point = {
            "pos": list(self.node.actual_pos),
            "spd_l": self.scale_left_spd.get(),
            "spd_r": self.scale_right_spd.get()
        }
        self.sequence.append(point)
        self.update_listbox()

    def update_listbox(self):
        self.listbox.delete(0, tk.END)
        for i, pt in enumerate(self.sequence):
            p = pt['pos']
            self.listbox.insert(tk.END, f"{i+1}: L({p[0]:.2f}, {p[1]:.2f}) R({p[2]:.2f}, {p[3]:.2f})")

    def delete_point(self):
        if self.listbox.curselection():
            del self.sequence[self.listbox.curselection()[0]]
            self.update_listbox()

    def clear_sequence(self):
        self.sequence = []
        self.update_listbox()

    def save_sequence(self):
        f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if f:
            with open(f, 'w') as file: json.dump(self.sequence, file)

    def load_sequence(self):
        f = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if f:
            with open(f, 'r') as file: self.sequence = json.load(file)
            self.update_listbox()

    def play_sequence(self):
        if not self.sequence: return
        threading.Thread(target=self._play_logic, daemon=True).start()

    def _play_logic(self):
        self.btn_play.configure(state="disabled")
        # Enable motors via parent App method if needed, or set directly
        self.node.motor_enabled = True 
        
        # Move to start
        start = self.sequence[0]['pos']
        self.node.set_targets_all(start, 2.0)
        
        # Lambda for checking if we are still running (Manual tab uses simple bool for now)
        is_active = lambda: True 
        
        if not wait_for_arrival(self.node, start, is_active):
             print("Error: Start Timeout")
             self.btn_play.configure(state="normal")
             return
        time.sleep(0.5)

        for i, step in enumerate(self.sequence):
            p = step['pos']
            sl = step['spd_l']
            sr = step['spd_r']
            
            # Set target with speeds
            self.node.set_target_arm(0, p[0], p[1], sl)
            self.node.set_target_arm(1, p[2], p[3], sr)
            
            wait_for_arrival(self.node, p, is_active)
            time.sleep(0.1)

        self.btn_play.configure(state="normal")

    def update_gui(self):
        # Update labels from node state
        p = self.node.actual_pos
        self.lbl_pos_left.configure(text=f"Actual: {p[0]:.2f}  {p[1]:.2f}")
        self.lbl_pos_right.configure(text=f"Actual: {p[2]:.2f}  {p[3]:.2f}")
        
        # Sync sliders if motors disabled (Follow mode)
        if not self.node.motor_enabled:
            self.scale_left_1.set(p[0])
            self.scale_left_2.set(p[1])
            self.scale_right_1.set(p[2])
            self.scale_right_2.set(p[3])

# --- TAB 2: ACTION BOARD ---
class ActionBoardTab:
    # CONFIG: Map incoming string messages to Action Slots
    TRIGGER_MAPPING = {
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
        "jab": 1, "hook": 2, "upper": 3
    }

    def __init__(self, parent, node):
        self.parent = parent
        self.node = node
        self.actions = {}
        self.action_queue = [] # Queue of IDs to play
        self.is_playing = False
        self.setup_ui()

    def setup_ui(self):
        # --- Top: Config & Queue Control ---
        top_frame = ttk.Frame(self.parent)
        top_frame.pack(fill="x", padx=10, pady=5)
        
        # Queue Mode Toggle
        self.var_queue_mode = tk.BooleanVar(value=False)
        chk_queue = ttk.Checkbutton(top_frame, text="Queue Mode", variable=self.var_queue_mode)
        chk_queue.pack(side="left", padx=10)
        
        # Auto-Run Toggle (NEW)
        self.var_auto_run = tk.BooleanVar(value=False)
        chk_auto = ttk.Checkbutton(top_frame, text="Auto-Run Queue", variable=self.var_auto_run)
        chk_auto.pack(side="left", padx=10)
        
        # Remote Trigger Toggle
        self.var_remote_enable = tk.BooleanVar(value=False)
        chk_remote = ttk.Checkbutton(top_frame, text="Remote Trigger", variable=self.var_remote_enable)
        chk_remote.pack(side="left", padx=10)
        
        # Speed Multiplier
        ttk.Label(top_frame, text="Speed:").pack(side="left", padx=(20, 5))
        self.scale_multiplier = tk.Scale(top_frame, from_=0.1, to=3.0, orient="horizontal", resolution=0.1)
        self.scale_multiplier.set(1.0)
        self.scale_multiplier.pack(side="left", fill="x", expand=True)

        # --- Main Split: Board (Left) vs Queue List (Right) ---
        paned = ttk.PanedWindow(self.parent, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=10, pady=5)
        
        board_frame = ttk.LabelFrame(paned, text="Action Board", padding=5)
        queue_frame = ttk.LabelFrame(paned, text="Playback Queue", padding=5)
        
        paned.add(board_frame, weight=3)
        paned.add(queue_frame, weight=1)

        # --- LEFT: BUTTON GRID ---
        for i in range(1, 7):
            slot_frame = ttk.LabelFrame(board_frame, text=f"Slot {i}", padding=5)
            r = (i-1) // 3
            c = (i-1) % 3
            slot_frame.grid(row=r, column=c, padx=5, pady=5, sticky="nsew")
            
            btn = tk.Button(slot_frame, text=f"ACTION {i}", bg="#dddddd", font=("Arial", 11, "bold"), height=2)
            btn.config(command=lambda idx=i: self.trigger_action(idx))
            btn.pack(fill="x", pady=2)
            
            btn_load = ttk.Button(slot_frame, text="Load...", command=lambda idx=i: self.load_file(idx))
            btn_load.pack(fill="x")
            
            lbl_file = ttk.Label(slot_frame, text="(Empty)", foreground="gray", anchor="center", width=15)
            lbl_file.pack(fill="x")
            
            if i not in self.actions:
                self.actions[i] = {'btn': btn, 'lbl': lbl_file, 'data': None}

        board_frame.columnconfigure(0, weight=1)
        board_frame.columnconfigure(1, weight=1)
        board_frame.columnconfigure(2, weight=1)
        board_frame.rowconfigure(0, weight=1)
        board_frame.rowconfigure(1, weight=1)

        # --- RIGHT: QUEUE LIST ---
        self.lst_queue = tk.Listbox(queue_frame, height=15)
        self.lst_queue.pack(fill="both", expand=True, pady=5)
        
        btn_run_q = tk.Button(queue_frame, text="▶ RUN QUEUE", bg="#aaffaa", font=("Arial", 10, "bold"), command=self.run_queue)
        btn_run_q.pack(fill="x", pady=2)
        
        btn_clear_q = ttk.Button(queue_frame, text="Clear Queue", command=self.clear_queue)
        btn_clear_q.pack(fill="x", pady=2)
        
        self.var_loop = tk.BooleanVar(value=False)
        chk_loop = ttk.Checkbutton(queue_frame, text="Loop Queue", variable=self.var_loop)
        chk_loop.pack(pady=5)

        # --- BOTTOM: STOP ---
        btn_stop = tk.Button(self.parent, text="STOP / DISABLE MOTORS", bg="red", fg="white", font=("Arial", 14, "bold"), command=self.emergency_stop)
        btn_stop.pack(fill="x", ipady=10, padx=10, pady=10)

    # --- LOGIC ---
    def load_file(self, idx):
        path = filedialog.askopenfilename(filetypes=[("JSON Sequence", "*.json")])
        if path:
            try:
                with open(path, 'r') as f: data = json.load(f)
                filename = os.path.basename(path)
                self.actions[idx]['data'] = data
                self.actions[idx]['lbl'].config(text=filename, foreground="black")
                self.actions[idx]['btn'].config(bg="#aaffaa")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def trigger_action(self, idx):
        data = self.actions[idx]['data']
        if not data:
            print(f"Slot {idx} is empty")
            return

        # Check mode
        if self.var_queue_mode.get():
            # Add to Queue
            self.action_queue.append(idx)
            self.update_queue_ui()
            
            # Auto-start if enabled
            if self.var_auto_run.get() and not self.is_playing:
                threading.Thread(target=self._queue_worker, daemon=True).start()
        else:
            # Play Immediately
            if self.is_playing: return
            threading.Thread(target=self._play_single, args=(data,), daemon=True).start()

    def update_queue_ui(self):
        self.lst_queue.delete(0, tk.END)
        for i, idx in enumerate(self.action_queue):
            name = self.actions[idx]['lbl'].cget("text")
            self.lst_queue.insert(tk.END, f"{i+1}. Action {idx}: {name}")

    def clear_queue(self):
        self.action_queue = []
        self.update_queue_ui()

    def run_queue(self):
        if self.is_playing: return
        if not self.action_queue: return
        threading.Thread(target=self._queue_worker, daemon=True).start()

    def emergency_stop(self):
        self.is_playing = False
        self.node.motor_enabled = False
        self.action_queue = [] # Clear pending
        self.update_queue_ui()
        print("EMERGENCY STOP")

    def update_logic(self):
        # Poll trigger
        msg = self.node.latest_trigger_msg
        if msg is not None:
            self.node.latest_trigger_msg = None
            if self.var_remote_enable.get():
                if msg in self.TRIGGER_MAPPING:
                    slot = self.TRIGGER_MAPPING[msg]
                    self.trigger_action(slot)
        
        # Watchdog for Auto-Run
        # If queue has items and nothing is playing, kickstart it
        if self.var_queue_mode.get() and self.var_auto_run.get():
            if self.action_queue and not self.is_playing:
                # Add tiny debounce logic or just start thread
                print("Auto-Run: Starting Queue Worker")
                threading.Thread(target=self._queue_worker, daemon=True).start()

    # --- PLAYBACK WORKERS ---
    
    def _execute_sequence_data(self, sequence):
        """Blocking function to execute one sequence"""
        self.node.motor_enabled = True
        multiplier = self.scale_multiplier.get()
        
        # Start Move
        start_pt = sequence[0]
        start_pos = parse_point(start_pt, self.node.actual_pos)
        self.node.set_targets_all(start_pos, 2.0 * multiplier)
        
        if not wait_for_arrival(self.node, start_pos, lambda: self.is_playing):
            return False # Cancelled
        
        time.sleep(0.5)
        
        for step in sequence:
            if not self.is_playing: return False
            
            targets = parse_point(step, self.node.actual_pos)
            raw_speeds = parse_speed(step)
            speeds = [s * multiplier for s in raw_speeds]
            
            self.node.set_targets_all(targets, speeds)
            wait_for_arrival(self.node, targets, lambda: self.is_playing)
            time.sleep(0.05)
            
        return True

    def _play_single(self, data):
        self.is_playing = True
        try:
            self._execute_sequence_data(data)
        finally:
            self.is_playing = False

    def _queue_worker(self):
        self.is_playing = True
        try:
            while self.is_playing:
                if not self.action_queue:
                    # Queue finished
                    if self.var_loop.get() and self.lst_queue.size() > 0:
                        # Reload initial queue? 
                        # Complex logic for looping infinite list. 
                        # For now, simple implementation: we consumed the list.
                        # To loop, we'd need to keep a copy.
                        # Let's just break for safety in V1.
                        break 
                    else:
                        break
                
                # Get next action
                idx = self.action_queue[0] # Peek
                data = self.actions[idx]['data']
                
                # Run it
                success = self._execute_sequence_data(data)
                
                if success:
                    # Remove from queue only if successful
                    self.action_queue.pop(0) 
                    # Update UI in thread-safe way? 
                    # Tkinter isn't thread safe, but usually tolerates simple calls.
                    # Ideally use after(), but here:
                    self.lst_queue.delete(0) 
                    time.sleep(0.5)
                else:
                    break # Stop if cancelled/error
                    
        finally:
            self.is_playing = False

# --- MAIN APP CONTAINER ---
class MainApp:
    def __init__(self, root, node):
        self.root = root
        self.node = node
        self.root.title("Dual Arm Control Center")
        self.root.geometry("1100x850")
        
        # Status Header
        header = ttk.Frame(root, padding=10)
        header.pack(fill="x")
        self.lbl_conn = ttk.Label(header, text="DISCONNECTED", foreground="red", font=("Arial", 12, "bold"))
        self.lbl_conn.pack(side="left")
        
        self.btn_master_enable = ttk.Button(header, text="SYSTEM ENABLE", command=self.enable_all)
        self.btn_master_enable.pack(side="right", padx=5)
        self.btn_master_disable = ttk.Button(header, text="SYSTEM DISABLE", command=self.disable_all)
        self.btn_master_disable.pack(side="right", padx=5)

        # Tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.tab1_frame = ttk.Frame(self.notebook)
        self.tab2_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab1_frame, text="Manual & Sequencer")
        self.notebook.add(self.tab2_frame, text="Action Board & Queue")
        
        # Initialize Tab Logic
        self.manual_tab = ManualTab(self.tab1_frame, node)
        self.action_tab = ActionBoardTab(self.tab2_frame, node)
        
        self.update_loop()

    def enable_all(self):
        self.node.motor_enabled = True
        self.node.target_pos = list(self.node.actual_pos)

    def disable_all(self):
        self.node.motor_enabled = False

    def update_loop(self):
        if time.time() - self.node.last_msg_time < 1.0:
            self.lbl_conn.config(text="CONNECTED", foreground="green")
        else:
            self.lbl_conn.config(text="DISCONNECTED", foreground="red")
            
        self.manual_tab.update_gui()
        self.action_tab.update_logic() 
        self.root.after(100, self.update_loop)

def main():
    rclpy.init()
    node = RobotNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    
    root = tk.Tk()
    app = MainApp(root, node)
    try:
        root.mainloop()
    finally:
        node.shutdown()
        rclpy.shutdown()
        spin_thread.join()

if __name__ == "__main__":
    main()