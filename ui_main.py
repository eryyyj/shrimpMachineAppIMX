import os
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from ui_biomass import BiomassWindow
from theme import *

# Fix scaling for Raspberry Pi 5 Touchscreens
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
os.environ["QT_SCALE_FACTOR"] = "1"
os.environ["QT_FONT_DPI"] = "96"
os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
os.environ.setdefault("QT_QPA_PLATFORM", "wayland")

class MainMenu(QtWidgets.QWidget):
    def __init__(self, user_id):
        super().__init__()
        self.user_id = user_id
        self.logout_requested = False

        # Window Configuration
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setFixedSize(1024, 600)
        
        # Background color
        self.setStyleSheet("background-color: #FAF7F2;") 

        # Main Layout
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(60, 50, 60, 60)
        self.main_layout.setSpacing(0)

        # 1. Middle Section: Welcome Text (Left) and Mascot (Right)
        self.mid_layout = QtWidgets.QHBoxLayout()
        self.mid_layout.setSpacing(0)
        
        # Left Side: "Welcome!" with a Fun Script Font
        self.lblWelcome = QtWidgets.QLabel("Welcome!")
        # We use a cursive/script font stack to mimic a fun hand-written style
        self.lblWelcome.setStyleSheet("""
            font-family: 'Comic Sans MS', 'Brush Script MT', 'Cursive';
            font-size: 110px; 
            font-weight: 500; 
            color: #0D3D45; 
            border: none;
        """)
        self.lblWelcome.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        # Right Side: Mascot Image
        self.lblImage = QtWidgets.QLabel()
        img_path = "/home/hiponpd/Documents/GitHub/ShrimpMachineApp/assets/images/landing.png"
        if os.path.exists(img_path):
            pixmap = QtGui.QPixmap(img_path).scaled(450, 450, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.lblImage.setPixmap(pixmap)
        self.lblImage.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        
        self.mid_layout.addWidget(self.lblWelcome, stretch=1)
        self.mid_layout.addWidget(self.lblImage, stretch=1)

        self.main_layout.addLayout(self.mid_layout, stretch=4)

        # 2. Bottom Section: Controls
        self.button_layout = QtWidgets.QHBoxLayout()

        # START Button
        self.btnStart = QtWidgets.QPushButton("START")
        self.btnStart.setFixedSize(280, 80)
        self.btnStart.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnStart.setStyleSheet("""
            QPushButton {
                background-color: #111111;
                color: white;
                border-radius: 20px; 
                font-size: 26px;
                font-weight: bold;
                letter-spacing: 2px;
            }
            QPushButton:pressed { background-color: #333333; }
        """)

        # Logout Link
        self.btnLogout = QtWidgets.QPushButton("logout")
        self.btnLogout.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnLogout.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #111111;
                font-size: 18px;
                font-weight: bold;
                border: none;
                text-decoration: underline;
            }
        """)

        self.button_layout.addWidget(self.btnStart)
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.btnLogout)

        self.main_layout.addLayout(self.button_layout, stretch=1)

        # Connections
        self.btnStart.clicked.connect(self.open_biomass)
        self.btnLogout.clicked.connect(self.logout)

    def open_biomass(self):
        self.bw = BiomassWindow(self.user_id, self)
        self.bw.showFullScreen()
        self.hide()

    def logout(self):
        self.logout_requested = True
        self.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # Force DPI settings for the Raspberry Pi display
    app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, False)
    app.setAttribute(QtCore.Qt.AA_DisableHighDpiScaling, True)
    app.setAttribute(QtCore.Qt.AA_Use96Dpi, True)

    window = MainMenu(user_id="test_user")
    window.showFullScreen()
    sys.exit(app.exec_())