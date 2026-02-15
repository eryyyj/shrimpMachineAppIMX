import sys, cv2, datetime, os, time 
from PyQt5 import QtWidgets, QtGui, QtCore
from compute import compute_feed
from detector import ShrimpDetector
from camera import Camera
from database import save_biomass_record
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
        self.detector = ShrimpDetector()
        self.camera = Camera()
        self.mqtt = MqttClient()
        self.mqtt.connect()
        
        self.running = False
        self.pump_on = False
        self.threshold_count = 0
        self.threshold_reached = False
        
        self.detect_enabled = True 

        self.prev_time = 0

        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setStyleSheet(f"background-color: {COLOR_BG}; color: {COLOR_TEAL};")
        self.showFullScreen()

        self.init_ui()
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

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
        
        self.lbl_status = QtWidgets.QLabel("SYSTEM READY")
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
        
        self.btn_secret = QtWidgets.QPushButton(self.btn_start)
        self.btn_secret.setGeometry(0, 0, 300, 25)  
        self.btn_secret.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_secret.setStyleSheet("background-color: transparent; border: none;")
        self.btn_secret.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_secret.clicked.connect(self.start_secret)

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
        self.timer.stop()
        self.detector.reset_total_count()
        self.threshold_count = 0
        self.threshold_reached = False
        self.lbl_count.setText("Count: 0")
        self.lbl_target.setText("Target: Not Set")
        self.lbl_status.setText("SYSTEM RESET / DOOR OPEN")
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

    def start_secret(self):
        self.detect_enabled = False # Disable detection
        self._start_common("RUNNING...")

    def _start_common(self, status_text):
        """ Shared logic for starting timer and UI updates """
        self.running = True
        self.prev_time = time.time()
        self.timer.start(100)
        self.lbl_status.setText(status_text)
        
        self.btn_stop.setEnabled(True)
        self.btn_stop.setStyleSheet(self.get_btn_style(COLOR_TEAL))
        self.btn_save.setEnabled(True)
        self.btn_save.setStyleSheet(self.get_btn_style(COLOR_TEAL))

    def stop(self):
        self.running = False
        self.timer.stop()
        self.lbl_status.setText("STOPPED")
        self.btn_dispense.setEnabled(True)
        self.btn_dispense.setStyleSheet(self.get_btn_style(COLOR_AQUA))

    def save(self):
        count = self.detector.total_count
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
        self.timer.stop()
        self.mqtt.disconnect()
        self.camera.release()
        if self.parent: self.parent.show()
        self.close()

    def set_count(self):
        dialog = NumberInputDialog(self)
        if dialog.exec_():
            num = dialog.get_number()
            if num > 0:
                self.threshold_count = num
                self.lbl_target.setText(f"Target: {num}")

    def update_frame(self):
        frame = self.camera.get_frame()
        if frame is not None:
            # --- FPS Calculation ---
            curr_time = time.time()
            fps = 0
            if self.prev_time > 0:
                fps = int(1.0 / (curr_time - self.prev_time))
            self.prev_time = curr_time

            if self.detect_enabled and not self.threshold_reached:
                # Normal mode: Run detection
                count, frame_rgb = self.detector.detect(frame)
                
                # Check threshold
                if self.threshold_count > 0 and count >= self.threshold_count:
                    self.mqtt.publish("shrimp/servo1/command", "SERVO1_CLOSE")
                    self.threshold_reached = True
                    self.lbl_status.setText("TARGET REACHED / DOOR CLOSED")
                
                # Update text stats
                self.lbl_count.setText(f"Count: {count}")
                b, f, p, fl = compute_feed(count)
                self.lbl_bio.setText(f"Biomass: {b:.2f}g\nFeed: {f:.2f}g")
                
            elif self.threshold_reached:
                # Target was reached: Keep showing video but STOP counting/detecting
                # We call detect with draw=False to just get the current count without processing
                count = self.detector.total_count
                
                # Just draw the count on the frame so the video looks "frozen" in time regarding data
                cv2.putText(frame, f"FINAL COUNT: {count}", (15, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            else:
                # Secret Mode / Detection Disabled
                h, w = frame.shape[:2]
                line_x = int(w * 0.3) 
                cv2.line(frame, (line_x, 0), (line_x, h), (0, 255, 255), 2)
                
                count = self.detector.total_count
                stats = f"FPS: {fps} | Count: {count}"
                
                (text_w, text_h), _ = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (10, 10), (10 + text_w + 10, 10 + text_h + 20), (0,0,0), -1)
                cv2.putText(frame, stats, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- Display logic is shared ---
            h, w, ch = frame_rgb.shape
            qimg = QtGui.QImage(frame_rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg).scaled(640, 420, QtCore.Qt.KeepAspectRatio)
            self.video_label.setPixmap(pix)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = BiomassWindow(user_id="test_user")
    sys.exit(app.exec_())