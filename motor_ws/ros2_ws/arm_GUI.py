#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import threading
import time
import json
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class ArmNode(Node):
    def __init__(self):
        super().__init__('dual_arm_gui')
        self.pub_cmd = self.create_publisher(Float64MultiArray, 'motor_commands', 10)
        self.create_subscription(Float64MultiArray, 'motor_feedback', self.feedback_callback, 10)
        
        # [Left1, Left2, Right1, Right2] -> [Motor 1, Motor 2, Motor 3, Motor 4]
        self.actual_pos = [0.0] * 4
        self.target_pos = [0.0] * 4
        self.target_speed = [5.0] * 4
        self.motor_enabled = False
        self.rx_count = 0
        self.last_msg_time = 0
        
        self.running = True
        self.thread = threading.Thread(target=self.heartbeat_loop, daemon=True)
        self.thread.start()

    def feedback_callback(self, msg):
        self.last_msg_time = time.time()
        if len(msg.data) >= 4:
            self.actual_pos = list(msg.data[0:4])
            if len(msg.data) >= 5:
                self.rx_count = int(msg.data[4])

    def heartbeat_loop(self):
        while self.running and rclpy.ok():
            msg = Float64MultiArray()
            # [P1, P2, P3, P4, S1, S2, S3, S4, Mode]
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

    def shutdown(self):
        self.running = False
        self.thread.join()

class App:
    def __init__(self, root, node):
        self.root = root
        self.node = node
        self.root.title("Damiao 4-Motor Controller")
        self.root.geometry("1000x800")
        self.sequence = [] 
        self.setup_ui()
        self.update_gui_loop()

    def setup_ui(self):
        # Top Status Bar
        status_frame = ttk.LabelFrame(self.root, text="System Status", padding=10)
        status_frame.pack(fill="x", padx=10, pady=5)
        self.lbl_conn = ttk.Label(status_frame, text="DISCONNECTED", foreground="red", font=("Arial", 12, "bold"))
        self.lbl_conn.pack(side="left")
        self.lbl_rx = ttk.Label(status_frame, text="RX: 0", font=("Arial", 12))
        self.lbl_rx.pack(side="right")
        
        # Global Enable
        btn_frame = ttk.Frame(status_frame)
        btn_frame.pack(side="top")
        self.btn_enable = ttk.Button(btn_frame, text="SYSTEM ENABLE (Stiff)", command=lambda: self.set_enabled(True))
        self.btn_enable.pack(side="left", padx=10)
        self.btn_disable = ttk.Button(btn_frame, text="SYSTEM DISABLE (Teach)", command=lambda: self.set_enabled(False))
        self.btn_disable.pack(side="left", padx=10)

        # Main Content Split
        main_pane = ttk.PanedWindow(self.root, orient="horizontal")
        main_pane.pack(fill="both", expand=True, padx=10, pady=5)

        # --- LEFT ARM FRAME (Motors 1 & 2) ---
        self.frame_left = ttk.LabelFrame(main_pane, text="Left Group", padding=10)
        main_pane.add(self.frame_left, weight=1)
        self.setup_arm_panel(self.frame_left, "Left", 0)

        # --- RIGHT ARM FRAME (Motors 3 & 4) ---
        self.frame_right = ttk.LabelFrame(main_pane, text="Right Group", padding=10)
        main_pane.add(self.frame_right, weight=1)
        self.setup_arm_panel(self.frame_right, "Right", 1)

        # --- SEQUENCER (Bottom) ---
        self.setup_sequencer()

    def setup_arm_panel(self, parent, name, idx):
        # Calculate Motor IDs based on arm index (0 or 1)
        id_a = (idx * 2) + 1
        id_b = (idx * 2) + 2

        # Readout
        lbl_pos = ttk.Label(parent, text=f"Actual: 0.00  0.00", font=("Courier", 12))
        lbl_pos.pack(pady=5)
        setattr(self, f"lbl_pos_{name.lower()}", lbl_pos)

        # Sliders
        s1 = tk.Scale(parent, from_=-12.5, to=12.5, orient="horizontal", label=f"Motor {id_a}", resolution=0.1)
        s1.pack(fill="x")
        setattr(self, f"scale_{name.lower()}_1", s1)
        
        s2 = tk.Scale(parent, from_=-12.5, to=12.5, orient="horizontal", label=f"Motor {id_b}", resolution=0.1)
        s2.pack(fill="x")
        setattr(self, f"scale_{name.lower()}_2", s2)
        
        spd = tk.Scale(parent, from_=0.5, to=30.0, orient="horizontal", label="Speed", resolution=0.5)
        spd.set(5.0)
        spd.pack(fill="x")
        setattr(self, f"scale_{name.lower()}_spd", spd)
        
        ttk.Button(parent, text=f"Move Motors {id_a} & {id_b}", command=lambda: self.send_arm(idx, name)).pack(pady=10)

    def setup_sequencer(self):
        seq_frame = ttk.LabelFrame(self.root, text="Global Sequencer (All 4 Motors)", padding=10)
        seq_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Tools
        tool_frame = ttk.Frame(seq_frame)
        tool_frame.pack(fill="x", pady=5)
        ttk.Button(tool_frame, text="Record Current Pose", command=self.record_point).pack(side="left", padx=2)
        ttk.Button(tool_frame, text="Delete Selected", command=self.delete_point).pack(side="left", padx=2)
        ttk.Button(tool_frame, text="Clear All", command=self.clear_sequence).pack(side="left", padx=2)
        ttk.Button(tool_frame, text="Save", command=self.save_sequence).pack(side="left", padx=10)
        ttk.Button(tool_frame, text="Load", command=self.load_sequence).pack(side="left", padx=2)

        # Play
        self.btn_play = ttk.Button(tool_frame, text="▶ PLAY ALL", command=self.play_sequence)
        self.btn_play.pack(side="right", padx=10)
        
        # List
        self.listbox = tk.Listbox(seq_frame, height=6)
        self.listbox.pack(fill="both", expand=True)
        
        self.lbl_status = ttk.Label(seq_frame, text="Ready", relief="sunken")
        self.lbl_status.pack(fill="x", pady=2)
        self.progress = ttk.Progressbar(seq_frame, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x")

    # --- LOGIC ---
    def set_enabled(self, state):
        self.node.motor_enabled = state
        if state:
            # Sync sliders to actual before enabling
            for i in range(4):
                self.node.target_pos[i] = self.node.actual_pos[i]
            # Update GUI sliders
            self.scale_left_1.set(self.node.actual_pos[0])
            self.scale_left_2.set(self.node.actual_pos[1])
            self.scale_right_1.set(self.node.actual_pos[2])
            self.scale_right_2.set(self.node.actual_pos[3])

    def send_arm(self, idx, name):
        # idx 0 = Left, 1 = Right
        s1 = getattr(self, f"scale_{name.lower()}_1").get()
        s2 = getattr(self, f"scale_{name.lower()}_2").get()
        spd = getattr(self, f"scale_{name.lower()}_spd").get()
        self.node.set_target_arm(idx, s1, s2, spd)

    def record_point(self):
        # Record all 4 motor positions + speeds
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
            self.listbox.insert(tk.END, f"{i+1}: M1({p[0]:.2f}) M2({p[1]:.2f}) | M3({p[2]:.2f}) M4({p[3]:.2f})")

    def delete_point(self):
        if self.listbox.curselection():
            del self.sequence[self.listbox.curselection()[0]]
            self.update_listbox()

    def clear_sequence(self):
        self.sequence = []
        self.update_listbox()

    def save_sequence(self):
        f = filedialog.asksaveasfilename(defaultextension=".json")
        if f:
            with open(f, 'w') as file: json.dump(self.sequence, file)

    def load_sequence(self):
        f = filedialog.askopenfilename()
        if f:
            with open(f, 'r') as file: self.sequence = json.load(file)
            self.update_listbox()

    def play_sequence(self):
        if self.sequence: threading.Thread(target=self._play_logic, daemon=True).start()

    def _play_logic(self):
        self.btn_play.configure(state="disabled")
        self.set_enabled(True)
        self.lbl_status.configure(text="Moving to Start...", background="yellow")
        
        # Move to start
        start = self.sequence[0]['pos']
        # Set all 4 targets
        for i in range(4): self.node.target_pos[i] = start[i]
        
        if not self.wait_for_arrival(start):
             self.lbl_status.configure(text="Error: Start Timeout", background="red")
             time.sleep(1)
             self.btn_play.configure(state="normal")
             return

        time.sleep(0.5)

        for i, step in enumerate(self.sequence):
            self.lbl_status.configure(text=f"Step {i+1}/{len(self.sequence)}", background="#aaf")
            self.progress['value'] = ((i+1)/len(self.sequence))*100
            
            # Set Targets
            p = step['pos']
            self.node.set_target_arm(0, p[0], p[1], step['spd_l'])
            self.node.set_target_arm(1, p[2], p[3], step['spd_r'])
            
            if not self.wait_for_arrival(p):
                print(f"Warning: Step {i+1} timeout")
            
            time.sleep(0.1)

        self.lbl_status.configure(text="Done", background="green")
        time.sleep(1.0)
        self.lbl_status.configure(text="Ready", background="white")
        self.btn_play.configure(state="normal")
        self.progress['value'] = 0

    def wait_for_arrival(self, targets):
        timeout = 8.0 
        start_time = time.time()
        while time.time() - start_time < timeout:
            all_arrived = True
            for i in range(4):
                if abs(self.node.actual_pos[i] - targets[i]) > 0.2:
                    all_arrived = False
                    break
            if all_arrived: return True
            time.sleep(0.05)
        return False

    def update_gui_loop(self):
        if time.time() - self.node.last_msg_time < 1.0:
            self.lbl_conn.configure(text="CONNECTED", foreground="green")
        else:
            self.lbl_conn.configure(text="DISCONNECTED", foreground="red")
        
        self.lbl_rx.configure(text=f"RX: {self.node.rx_count}")
        
        # Update Labels
        p = self.node.actual_pos
        self.lbl_pos_left.configure(text=f"Actual: M1={p[0]:.2f}  M2={p[1]:.2f}")
        self.lbl_pos_right.configure(text=f"Actual: M3={p[2]:.2f}  M4={p[3]:.2f}")
        
        self.root.after(100, self.update_gui_loop)

def main():
    rclpy.init()
    node = ArmNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    
    root = tk.Tk()
    app = App(root, node)
    try:
        root.mainloop()
    finally:
        node.shutdown()
        rclpy.shutdown()
        spin_thread.join()

if __name__ == "__main__":
    main()