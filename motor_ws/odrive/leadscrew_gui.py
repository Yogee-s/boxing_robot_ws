import tkinter as tk
from tkinter import ttk
import sys
import odrive
from odrive.enums import *

# --- CONFIGURATION ---
CURRENT_LIMIT = 4.0         # Amps
MAX_SPEED_RPS = 80.0        # Max speed in Turns/Second (40 t/s = 2400 RPM)
RAMP_RATE = 40           # Acceleration
ERROR_CHECK_RATE = 50       # Update rate (ms)

# DIRECTION SETTING
# Set to 1.0 or -1.0 to flip the "Up Arrow" direction
DIRECTION_FLIP = -1.0 

class FinalGUI:
    def __init__(self, root, axis):
        self.root = root
        self.axis = axis
        self.root.title("GIM6010 Lead Screw Controller")
        self.root.geometry("500x450")
        
        self.target_vel = 0.0
        self.is_moving = False
        self.glitch_count = 0

        # --- STYLE ---
        style = ttk.Style()
        style.configure("Big.TLabel", font=("Consolas", 12))
        style.configure("Header.TLabel", font=("Arial", 10, "bold"))

        # --- 1. MONITORING SECTION ---
        status_frame = ttk.LabelFrame(root, text="Live Telemetry", padding=10)
        status_frame.pack(fill="x", padx=10, pady=5)

        # Row 1: Position & Status
        self.lbl_status = ttk.Label(status_frame, text="IDLE", foreground="orange", font=("Arial", 12, "bold"))
        self.lbl_status.pack(anchor="w")
        
        self.lbl_pos = ttk.Label(status_frame, text="Position: 0.00 revs", style="Big.TLabel")
        self.lbl_pos.pack(anchor="w")

        # Row 2: Vital Stats (Current, Temp, RPM)
        stats_frame = ttk.Frame(status_frame)
        stats_frame.pack(fill="x", pady=5)
        
        self.lbl_current = ttk.Label(stats_frame, text="Cur: 0.0 A", style="Big.TLabel", width=15)
        self.lbl_current.pack(side="left")
        
        self.lbl_temp = ttk.Label(stats_frame, text="Tmp: 00 C", style="Big.TLabel", width=15)
        self.lbl_temp.pack(side="left")

        self.lbl_rpm = ttk.Label(status_frame, text="Motor RPM: 0", style="Big.TLabel", foreground="blue")
        self.lbl_rpm.pack(anchor="w", pady=5)

        # Error Counter
        self.lbl_errors = ttk.Label(status_frame, text="Glitches Caught: 0", foreground="red")
        self.lbl_errors.pack(anchor="w")

        # --- 2. CONTROL SECTION ---
        control_frame = ttk.LabelFrame(root, text="Manual Drive", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(control_frame, text="Jog Speed Adjustment:", style="Header.TLabel").pack(anchor="w")
        self.speed_slider = ttk.Scale(control_frame, from_=1.0, to=MAX_SPEED_RPS, orient="horizontal")
        self.speed_slider.set(10.0) 
        self.speed_slider.pack(fill="x", pady=5)

        # Instructions
        help_frame = ttk.Frame(control_frame)
        help_frame.pack(pady=10)
        ttk.Label(help_frame, text="HOLD UP", font=("Arial", 10, "bold")).pack(side="top")
        ttk.Label(help_frame, text="to Extend").pack(side="top")
        ttk.Label(help_frame, text="-----------").pack(side="top")
        ttk.Label(help_frame, text="HOLD DOWN", font=("Arial", 10, "bold")).pack(side="top")
        ttk.Label(help_frame, text="to Retract").pack(side="top")
        
        # Checkbox
        self.retry_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Auto-Clear Errors (Anti-Glitch)", variable=self.retry_var).pack(pady=5)

        self.btn_stop = ttk.Button(root, text="EMERGENCY STOP (SPACE)", command=self.emergency_stop)
        self.btn_stop.pack(fill="x", padx=10, pady=10, ipady=10)

        # --- BINDINGS ---
        root.bind('<KeyPress-Up>', lambda e: self.set_vel(self.speed_slider.get()))
        root.bind('<KeyRelease-Up>', lambda e: self.set_vel(0))
        root.bind('<KeyPress-Down>', lambda e: self.set_vel(-self.speed_slider.get()))
        root.bind('<KeyRelease-Down>', lambda e: self.set_vel(0))
        root.bind('<space>', lambda e: self.emergency_stop())

        # Start Loop
        self.watchdog_loop()

    def set_vel(self, vel):
        # Apply Direction Flip
        final_vel = vel * DIRECTION_FLIP
        
        self.target_vel = final_vel
        
        # Auto-Engage Closed Loop if needed
        if self.axis.current_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
             self.axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        
        self.axis.controller.input_vel = final_vel
        
        # Update Status Text
        self.is_moving = (vel != 0)
        if vel > 0:
            self.lbl_status.config(text="EXTENDING >>", foreground="green")
        elif vel < 0:
            self.lbl_status.config(text="<< RETRACTING", foreground="blue")
        else:
            self.lbl_status.config(text="HOLDING", foreground="black")

    def emergency_stop(self):
        self.axis.requested_state = AXIS_STATE_IDLE
        self.target_vel = 0
        self.is_moving = False
        self.lbl_status.config(text="STOPPED", foreground="red")

    def watchdog_loop(self):
        try:
            # 1. Error Handling (The Anti-Glitch Logic)
            if self.axis.error != 0:
                axis_err = self.axis.error
                enc_err = self.axis.encoder.error
                
                # Log it
                self.glitch_count += 1
                self.lbl_errors.config(text=f"Last: {hex(axis_err)} (Total: {self.glitch_count})")
                
                if self.retry_var.get():
                    self.axis.clear_errors()
                    # Re-engage if button is held
                    if self.is_moving:
                        self.axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
                        self.axis.controller.input_vel = self.target_vel

            # 2. Read Telemetry
            pos = self.axis.encoder.pos_estimate
            current = self.axis.motor.current_control.Iq_measured
            temp = self.axis.fet_thermistor.temperature
            vel_rps = self.axis.encoder.vel_estimate # Turns per second
            rpm = vel_rps * 60.0

            # 3. Update Labels
            self.lbl_pos.config(text=f"Position: {pos:.2f} revs")
            self.lbl_current.config(text=f"Cur: {current:.1f} A")
            self.lbl_temp.config(text=f"Tmp: {temp:.0f} °C")
            self.lbl_rpm.config(text=f"Motor RPM: {int(rpm)}")

        except Exception as e:
            pass # Ignore read errors (e.g. USB packet loss)
            
        self.root.after(ERROR_CHECK_RATE, self.watchdog_loop)

def main():
    print("Connecting to ODrive...")
    odrv0 = odrive.find_any()
    axis = odrv0.axis0

    # Ensure Software Filter is applied (Lowers bandwidth to ignore noise)
    print("Applying Noise Filter (Bandwidth = 500)...")
    axis.encoder.config.bandwidth = 500
    
    # Configure Limits
    axis.controller.config.vel_limit = MAX_SPEED_RPS * 1.5
    axis.motor.config.current_lim = CURRENT_LIMIT
    
    # Configure Mode
    axis.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
    axis.controller.config.input_mode = INPUT_MODE_VEL_RAMP
    axis.controller.config.vel_ramp_rate = RAMP_RATE
    
    print("Launching Dashboard...")
    root = tk.Tk()
    app = FinalGUI(root, axis)
    root.mainloop()

    # Safety on exit
    print("Closing... Disengaging Motor.")
    axis.requested_state = AXIS_STATE_IDLE

if __name__ == "__main__":
    main()
