import sys
import cv2
import datetime
import time
from PyQt5 import QtWidgets, QtGui, QtCore
from compute import compute_feed
from database import save_biomass_record

try:
    from imx500_camera import IMX500Camera, IMX500Worker
    IMX500_AVAILABLE = True
except Exception:
    IMX500_AVAILABLE = False

from mqtt_client import MqttClient

# --- Aquaculture Palette ---
COLOR_BG = "#FAF7F2"
COLOR_TEAL = "#0D3D45"   # Primary
COLOR_AQUA = "#2A9D8F"   # Active
COLOR_NEUTRAL = "#E0E0E0" # Disabled/Grey

class NumberInputDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setStyleSheet(f"background-color:{COLOR_BG}; border: 2px solid {COLOR_TEAL}; border-radius: 15px;")
        self.setModal(True)
        self.setMinimumWidth(350)
        self.current_value = "0"
        
        layout = QtWidgets.QVBoxLayout(self)
        self.display = QtWidgets.QLabel("0")
        self.display.setStyleSheet(f"font-size: 40px; font-weight: bold; background: white; color: {COLOR_TEAL}; padding: 20px; border-radius: 10px;")
        self.display.setAlignment(QtCore.Qt.AlignRight)
        layout.addWidget(self.display)

        grid = QtWidgets.QGridLayout()
        buttons = ['7', '8', '9', '4', '5', '6', '1', '2', '3', 'Clear', '0', 'OK']
        for i, name in enumerate(buttons):
            btn = QtWidgets.QPushButton(name)
            btn.setFixedSize(80, 60)
            btn.setStyleSheet(f"background-color: white; color: {COLOR_TEAL}; font-size: 20px; font-weight: bold; border-radius: 10px;")
            if name == 'OK': btn.clicked.connect(self.accept)
            elif name == 'Clear': btn.clicked.connect(self.clear)
            else: btn.clicked.connect(lambda ch, n=name: self.append_num(n))
            grid.addWidget(btn, i//3, i%3)
        layout.addLayout(grid)

    def append_num(self, n):
        self.current_value = n if self.current_value == "0" else self.current_value + n
        self.display.setText(self.current_value)

    def clear(self):
        self.current_value = "0"
        self.display.setText("0")

    def get_number(self):
        return int(self.current_value)
    
class BiomassWindow(QtWidgets.QWidget):
    def __init__(self, user_id, parent=None):
        super().__init__()
        self.user_id = user_id
        self.parent = parent
        self.mqtt = MqttClient()
        self.mqtt.connect()

        # IMX500 camera and worker
        self.imx500_camera = None
        self.imx500_worker = None
        if IMX500_AVAILABLE:
            try:
                self.imx500_camera = IMX500Camera()
                self.imx500_worker = IMX500Worker(self.imx500_camera)
                self.imx500_worker.frame_ready.connect(self.on_frame_ready)
            except Exception:
                self.imx500_camera = None
                self.imx500_worker = None

        self.running = False
        self.pump_on = False
        self.threshold_count = 0
        self.threshold_reached = False
        self.detect_enabled = True
        self.current_count = 0
        self.prev_time = 0

        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setStyleSheet(f"background-color: {COLOR_BG}; color: {COLOR_TEAL};")
        self.showFullScreen()

        self.init_ui()

    def init_ui(self):
        # Overall vertical layout
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 10, 20, 20) 
        self.main_layout.setSpacing(10)

        # 1. TOP BAR
        top_bar = QtWidgets.QHBoxLayout()
        self.btn_back = QtWidgets.QPushButton()
        self.btn_back.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowLeft))
        self.btn_back.setIconSize(QtCore.QSize(35, 35))
        self.btn_back.setFixedSize(50, 50)
        self.btn_back.setFlat(True)
        self.btn_back.clicked.connect(self.go_back)

        self.lbl_title = QtWidgets.QLabel("BIOMASS CALCULATION")
        self.lbl_title.setStyleSheet(f"font-size: 28px; font-weight: 900; color: {COLOR_TEAL}; letter-spacing: 2px;")
        self.lbl_title.setAlignment(QtCore.Qt.AlignCenter)

        top_bar.addWidget(self.btn_back)
        top_bar.addStretch()
        top_bar.addWidget(self.lbl_title)
        top_bar.addStretch()
        top_bar.addSpacing(50)
        self.main_layout.addLayout(top_bar)

        # 2. CONTENT AREA
        content_hbox = QtWidgets.QHBoxLayout()
        content_hbox.setSpacing(30)

        # --- LEFT SIDE: CAMERA ---
        left_layout = QtWidgets.QVBoxLayout()
        self.video_label = QtWidgets.QLabel()
        self.video_label.setStyleSheet(f"background-color: black; border-radius: 15px; border: 2px solid {COLOR_TEAL};")
        self.video_label.setFixedSize(640, 420)
        
        self.lbl_status = QtWidgets.QLabel(
            "CAMERA UNAVAILABLE" if self.imx500_camera is None else "SYSTEM READY"
        )
        self.lbl_status.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {COLOR_AQUA};")
        
        left_layout.addWidget(self.video_label)
        left_layout.addWidget(self.lbl_status)
        content_hbox.addLayout(left_layout)

        # --- RIGHT SIDE: DATA & CONTROLS ---
        right_layout = QtWidgets.QVBoxLayout()
        
        data_container = QtWidgets.QFrame()
        data_container.setStyleSheet(f"background: white; border-radius: 15px; border: 2px solid {COLOR_NEUTRAL};")
        data_vbox = QtWidgets.QVBoxLayout(data_container)
        
        self.lbl_target = QtWidgets.QLabel("Target: Not Set")
        self.lbl_target.setStyleSheet("font-size: 13px; color: #555; border: none; font-weight: bold;")
        self.lbl_count = QtWidgets.QLabel("Count: 0")
        self.lbl_count.setStyleSheet(f"font-size: 36px; font-weight: 900; color: {COLOR_TEAL}; border: none;")
        self.lbl_bio = QtWidgets.QLabel("Biomass: 0.00g\nFeed: 0.00g")
        self.lbl_bio.setStyleSheet("font-size: 18px; color: #444; border: none; line-height: 140%;")
        
        data_vbox.addWidget(self.lbl_target)
        data_vbox.addWidget(self.lbl_count)
        data_vbox.addWidget(self.lbl_bio)
        
        # Button Grid
        btn_grid = QtWidgets.QGridLayout()
        btn_grid.setSpacing(10)
        
        self.btn_set = self.create_btn("SET TARGET", COLOR_TEAL)
        self.btn_start = self.create_btn("START", COLOR_TEAL)
        if self.imx500_camera is None:
            self.btn_start.setEnabled(False)
        
        self.btn_serviceoverlay = QtWidgets.QPushButton(self.btn_start)
        self.btn_serviceoverlay.setGeometry(0, 0, 300, 25)  
        self.btn_serviceoverlay.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_serviceoverlay.setStyleSheet("background-color: transparent; border: none;")
        self.btn_serviceoverlay.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_serviceoverlay.clicked.connect(self.start_serviceoverlay)

        self.btn_stop = self.create_btn("STOP", COLOR_NEUTRAL)
        self.btn_stop.setEnabled(False)
        self.btn_save = self.create_btn("SAVE", COLOR_NEUTRAL)
        self.btn_save.setEnabled(False)
        self.btn_dispense = self.create_btn("DISPENSE FEED", COLOR_NEUTRAL)
        self.btn_dispense.setEnabled(False)
        
        self.btn_pump = self.create_btn("PUMP: OFF", COLOR_TEAL)
        self.btn_reset = self.create_btn("RESET", COLOR_NEUTRAL)

        btn_grid.addWidget(self.btn_set, 0, 0)
        btn_grid.addWidget(self.btn_start, 0, 1)
        btn_grid.addWidget(self.btn_stop, 1, 0)
        btn_grid.addWidget(self.btn_pump, 1, 1)
        btn_grid.addWidget(self.btn_save, 2, 0)
        btn_grid.addWidget(self.btn_reset, 2, 1)
        btn_grid.addWidget(self.btn_dispense, 3, 0, 1, 2)

        right_layout.addWidget(data_container)
        right_layout.addSpacing(15)
        right_layout.addLayout(btn_grid)
        right_layout.addStretch()
        
        content_hbox.addLayout(right_layout)
        self.main_layout.addLayout(content_hbox)

        # Bindings
        self.btn_set.clicked.connect(self.set_count)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_pump.clicked.connect(self.toggle_pump)
        self.btn_reset.clicked.connect(self.reset_all)
        self.btn_save.clicked.connect(self.save)
        self.btn_dispense.clicked.connect(self.dispense)

    def create_btn(self, text, color):
        btn = QtWidgets.QPushButton(text)
        btn.setFixedHeight(60)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        btn.setStyleSheet(self.get_btn_style(color))
        return btn

    def get_btn_style(self, color):
        txt = "white" if color != COLOR_NEUTRAL else "#555"
        return f"background-color: {color}; color: {txt}; border-radius: 10px; font-weight: bold; font-size: 13px;"

    def toggle_pump(self):
        self.pump_on = not self.pump_on
        state = "ON" if self.pump_on else "OFF"
        self.btn_pump.setText(f"PUMP: {state}")
        color = COLOR_AQUA if self.pump_on else COLOR_TEAL
        self.btn_pump.setStyleSheet(self.get_btn_style(color))
        self.mqtt.publish("shrimp/pump/command", f"PUMP {state}")

    def reset_all(self):
        self.running = False
        if self.imx500_worker and self.imx500_worker.isRunning():
            self.imx500_worker.request_stop()
            self.imx500_worker.wait(2000)
        if self.imx500_camera:
            self.imx500_camera.reset_count()
        self.current_count = 0
        self.threshold_count = 0
        self.threshold_reached = False
        self.lbl_count.setText("Count: 0")
        self.lbl_target.setText("Target: Not Set")
        self.lbl_status.setText("SYSTEM RESET")
        self.mqtt.publish("shrimp/servo1/command", "SERVO1_OPEN")

        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(self.get_btn_style(COLOR_NEUTRAL))
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet(self.get_btn_style(COLOR_NEUTRAL))
        self.btn_dispense.setEnabled(False)
        self.btn_dispense.setStyleSheet(self.get_btn_style(COLOR_NEUTRAL))

    def start(self):
        """ Normal Start (Detection Enabled) """
        self.detect_enabled = True # Enable detection
        self._start_common("RUNNING...")

    def start_serviceoverlay(self):
        self.detect_enabled = False
        self._start_common("RUNNING...")

    def _start_common(self, status_text):
        """Shared logic for starting camera worker."""
        if self.imx500_camera is None:
            self.lbl_status.setText("CAMERA UNAVAILABLE")
            return
        self.running = True
        self.prev_time = time.time()
        self.imx500_worker._stop_requested = False
        self.imx500_worker.start()
        self.lbl_status.setText(status_text)

        self.btn_stop.setEnabled(True)
        self.btn_stop.setStyleSheet(self.get_btn_style(COLOR_TEAL))
        self.btn_save.setEnabled(True)
        self.btn_save.setStyleSheet(self.get_btn_style(COLOR_TEAL))

    def stop(self):
        self.running = False
        if self.imx500_worker and self.imx500_worker.isRunning():
            self.imx500_worker.request_stop()
            self.imx500_worker.wait(2000)
        self.lbl_status.setText("STOPPED")
        self.btn_dispense.setEnabled(True)
        self.btn_dispense.setStyleSheet(self.get_btn_style(COLOR_AQUA))

    def save(self):
        count = self.current_count
        b, f, p, fl = compute_feed(count)
        save_biomass_record(self.user_id, count, b, f)
        
        import json
        payload = {
            "userId": self.user_id,
            "shrimpCount": count,
            "biomass": round(b, 2),
            "feed": round(f, 2),
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.mqtt.publish(f"shrimp/updates/{self.user_id}", json.dumps(payload))
        
        from database import sync_biomass_records
        sync_biomass_records(self.user_id)
        
        self.lbl_status.setText("DATA SENT TO CLOUD & MOBILE")
        self.btn_dispense.setEnabled(True)
        self.btn_dispense.setStyleSheet(self.get_btn_style(COLOR_AQUA))

    def dispense(self):
        self.mqtt.publish("shrimp/servo3/command", "SERVO3_DISPENSE")
        self.lbl_status.setText("FEED DISPENSED")

    def go_back(self):
        if self.imx500_worker and self.imx500_worker.isRunning():
            self.imx500_worker.request_stop()
            self.imx500_worker.wait(2000)
        if self.imx500_camera:
            self.imx500_camera.stop()
        self.mqtt.disconnect()
        if self.parent:
            self.parent.show()
        self.close()

    def set_count(self):
        dialog = NumberInputDialog(self)
        if dialog.exec_():
            num = dialog.get_number()
            if num > 0:
                self.threshold_count = num
                self.lbl_target.setText(f"Target: {num}")

    def on_frame_ready(self, frame, count):
        """Handle frame and count from IMX500 worker."""
        self.current_count = count

        if frame is None:
            self.video_label.setText("No frame")
            return

        # Threshold check when detection enabled
        if self.detect_enabled and not self.threshold_reached:
            if self.threshold_count > 0 and count >= self.threshold_count:
                self.mqtt.publish("shrimp/servo1/command", "SERVO1_CLOSE")
                self.threshold_reached = True
                self.lbl_status.setText("TARGET REACHED")

        # Update display
        self.lbl_count.setText(f"Count: {count}")
        b, f, p, fl = compute_feed(count)
        self.lbl_bio.setText(f"Biomass: {b:.2f}g\nFeed: {f:.2f}g")

        # Overlay "FINAL COUNT" when target reached
        if self.threshold_reached:
            cv2.putText(
                frame, f"FINAL COUNT: {count}", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

        # Convert to RGB for display (picamera2 may return BGRA)
        if frame.shape[2] == 4:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = frame_rgb.shape
        qimg = QtGui.QImage(frame_rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(640, 420, QtCore.Qt.KeepAspectRatio)
        self.video_label.setPixmap(pix)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = BiomassWindow(user_id="test_user")
    sys.exit(app.exec_())