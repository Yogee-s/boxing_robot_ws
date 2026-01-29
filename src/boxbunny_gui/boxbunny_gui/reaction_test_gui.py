import sys
import signal
try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    print("Please install PySide6: pip install PySide6")
    sys.exit(1)

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from boxbunny_msgs.srv import StartStopDrill

APP_STYLESHEET = """
QWidget { background-color: #111317; color: #E6E6E6; font-family: 'DejaVu Sans'; font-size: 16px; }
QPushButton { background-color: #2ea043; border: 1px solid #394151; padding: 15px; border-radius: 8px; font-weight: bold; }
QPushButton:hover { background-color: #3fb950; }
QPushButton:pressed { background-color: #238636; }
QPushButton#stop { background-color: #da3633; }
QPushButton#stop:hover { background-color: #f85149; }
QLabel { color: #E6E6E6; }
"""

class ReactionTestNode(Node):
    def __init__(self, signals):
        super().__init__("reaction_test_gui")
        self.signals = signals
        
        self.create_subscription(String, "drill_state", self._on_state, 10)
        self.create_subscription(Int32, "drill_countdown", self._on_countdown, 10)
        self.create_subscription(String, "drill_summary", self._on_summary, 10)
        
        self.cli = self.create_client(StartStopDrill, "start_stop_drill")
        
    def start_drill(self):
        if not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("Drill service not available")
            return
        req = StartStopDrill.Request()
        req.start = True
        req.num_trials = 5
        self.cli.call_async(req)
        
    def stop_drill(self):
        if not self.cli.service_is_ready():
            return
        req = StartStopDrill.Request()
        req.start = False
        self.cli.call_async(req)

    def _on_state(self, msg):
        self.signals.state.emit(msg.data)

    def _on_countdown(self, msg):
        self.signals.countdown.emit(msg.data)

    def _on_summary(self, msg):
        self.signals.summary.emit(msg.data)


class RosWorker(QtCore.QThread):
    state = QtCore.Signal(str)
    countdown = QtCore.Signal(int)
    summary = QtCore.Signal(str)
    
    def __init__(self):
        super().__init__()
        self.node = None
        
    def run(self):
        rclpy.init()
        self.node = ReactionTestNode(self)
        try:
            rclpy.spin(self.node)
        except KeyboardInterrupt:
            pass
        finally:
            self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
                
    def start_drill(self):
        if self.node:
            self.node.start_drill()
            
    def stop_drill(self):
        if self.node:
            self.node.stop_drill()

class ReactionTestGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reaction Drill Test")
        self.resize(400, 300)
        
        self.worker = RosWorker()
        self.worker.state.connect(self._update_state)
        self.worker.countdown.connect(self._update_countdown)
        self.worker.summary.connect(self._update_summary)
        self.worker.start()
        
        self._build_ui()
        
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Status
        self.lbl_state = QtWidgets.QLabel("Status: IDLE")
        self.lbl_state.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_state.setStyleSheet("font-size: 18px; color: #8b949e;")
        layout.addWidget(self.lbl_state)
        
        # Countdown
        self.lbl_countdown = QtWidgets.QLabel("--")
        self.lbl_countdown.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_countdown.setStyleSheet("font-size: 64px; font-weight: bold; color: #58a6ff;")
        layout.addWidget(self.lbl_countdown)
        
        # Controls
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("START DRILL")
        self.btn_start.clicked.connect(self.worker.start_drill)
        
        self.btn_stop = QtWidgets.QPushButton("STOP")
        self.btn_stop.setObjectName("stop")
        self.btn_stop.clicked.connect(self.worker.stop_drill)
        
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)
        
        # Summary
        self.lbl_summary = QtWidgets.QLabel("Waiting for results...")
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_summary.setStyleSheet("font-size: 12px; color: #8b949e;")
        layout.addWidget(self.lbl_summary)

    def _update_state(self, state):
        self.lbl_state.setText(f"Status: {state.upper()}")
        if state == "cue":
            self.lbl_countdown.setText("PUNCH!")
            self.lbl_countdown.setStyleSheet("font-size: 64px; font-weight: bold; color: #ff7b72;")
        elif state == "waiting":
             self.lbl_countdown.setText("Wait...")
             self.lbl_countdown.setStyleSheet("font-size: 48px; color: #e3b341;")

    def _update_countdown(self, count):
        self.lbl_countdown.setText(str(count))
        self.lbl_countdown.setStyleSheet("font-size: 64px; font-weight: bold; color: #58a6ff;")

    def _update_summary(self, data):
        self.lbl_summary.setText(f"Last Drill Summary: {data}")

    def closeEvent(self, event):
        self.worker.quit()
        self.worker.wait()
        event.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    gui = ReactionTestGui()
    gui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
