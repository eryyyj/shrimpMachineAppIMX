import sys
import os
import qrcode
from PyQt5 import QtWidgets, QtCore, QtGui
from database import init_db, create_qr_session, poll_for_login, verify_user_credentials
from ui_main import MainMenu

# --- Environment setup for Wayland/RPi5 ---
os.environ.setdefault("QT_QPA_PLATFORM", "wayland")

class Login(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ShrimpSense Login")
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setStyleSheet("background-color: #FAF7F2;") 
        
        self.session_id = create_qr_session()
        self.user_id = None

        # Create the "Remember Me" checkbox once
        self.cb_remember = QtWidgets.QCheckBox("Stay logged in")
        self.cb_remember.setChecked(True)
        self.cb_remember.setStyleSheet("""
            QCheckBox { font-size: 16px; color: #0D3D45; font-weight: bold; border: none; }
            QCheckBox::indicator { width: 22px; height: 22px; }
        """)

        # Use a Stacked Widget to switch between QR and Manual Login
        self.stack = QtWidgets.QStackedWidget()
        self.setup_qr_view()
        self.setup_manual_view()
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.stack)

        # Polling Timer for QR
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.check_login_status)
        self.timer.start(2000)

    def setup_qr_view(self):
        qr_widget = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QVBoxLayout(qr_widget) 
        self.main_layout.setContentsMargins(100, 40, 100, 80) # Adjusted top margin for button
        self.main_layout.setSpacing(20)

        # --- TOP BAR: Back Button (Exit) ---
        self.btn_back = QtWidgets.QPushButton(qr_widget)
        self.btn_back.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowLeft))
        self.btn_back.setIconSize(QtCore.QSize(35, 35))
        self.btn_back.setFixedSize(50, 50)
        self.btn_back.setFlat(True)
        self.btn_back.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_back.move(20, 20) # Positions it in the absolute top-left corner
        self.btn_back.clicked.connect(self.reject)        

        # 1. Logo
        logo_layout = QtWidgets.QHBoxLayout()
        self.logo = QtWidgets.QLabel()
        logo_path = "/home/hiponpd/Documents/GitHub/ShrimpMachineApp/assets/images/ShrimpSenseLogo.png"
        if os.path.exists(logo_path):
            logo_pix = QtGui.QPixmap(logo_path).scaled(120, 120, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.logo.setPixmap(logo_pix)
        logo_layout.addWidget(self.logo)
        logo_layout.addStretch()
        self.main_layout.addLayout(logo_layout) # Updated to self.main_layout

        self.main_layout.addStretch()

        # 2. QR Card
        card = QtWidgets.QFrame()
        card.setStyleSheet("QFrame { background-color: white; border: 1px solid #D1D1D1; border-radius: 20px; }")
        card_layout = QtWidgets.QHBoxLayout(card)
        card_layout.setContentsMargins(40, 40, 40, 40)
        card_layout.setSpacing(40)

        text_container = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("Steps to log in")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #111; border: none;")
        steps = QtWidgets.QLabel(
            "1. Open <b>ShrimpSense</b> app on your phone.<br><br>"
            "2. Go to <b>Scan → Start Scanner.</b><br><br>"
            "3. Scan the QR code displayed on this screen."
        )
        steps.setStyleSheet("font-size: 16px; color: #333; border: none;")
        text_container.addWidget(title)
        text_container.addWidget(steps)
        text_container.addStretch()

        self.qr_label = QtWidgets.QLabel()
        self.qr_label.setStyleSheet("border: none;")
        if self.session_id:
            qr = qrcode.QRCode(version=1, box_size=6, border=2)
            qr.add_data(self.session_id)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            img.save("session_qr.png")
            self.qr_label.setPixmap(QtGui.QPixmap("session_qr.png").scaled(240, 240))
        
        card_layout.addLayout(text_container, stretch=2)
        card_layout.addWidget(self.qr_label, stretch=1)
        self.main_layout.addWidget(card)

        self.main_layout.addWidget(self.cb_remember, alignment=QtCore.Qt.AlignCenter)

        # 3. "Try another way" Link
        self.btn_switch = QtWidgets.QPushButton("Try another way")
        self.btn_switch.setStyleSheet("""
            QPushButton { 
                color: #0D3D45; font-size: 12px; font-weight: bold; 
                text-decoration: underline; background: transparent; border: none;
            }
        """)
        self.btn_switch.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_switch.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        self.main_layout.addWidget(self.btn_switch, alignment=QtCore.Qt.AlignCenter)
        
        self.main_layout.addStretch()
        self.stack.addWidget(qr_widget)

    def setup_manual_view(self):
        manual_widget = QtWidgets.QWidget()
        # Full screen layout
        layout = QtWidgets.QVBoxLayout(manual_widget)
        layout.setContentsMargins(100, 40, 100, 80)

        # --- UNDER DEVELOPMENT SIGN ---
        self.ud_screen_label = QtWidgets.QLabel(manual_widget)
        ud_path = "/home/hiponpd/Documents/GitHub/ShrimpMachineApp/assets/images/ud.png"
        
        if os.path.exists(ud_path):
            ud_pix = QtGui.QPixmap(ud_path)
            
            scaled_pix = ud_pix.scaled(250, 250, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.ud_screen_label.setPixmap(scaled_pix)

            self.ud_screen_label.setFixedSize(300, 300)
            self.ud_screen_label.move(770, -15)
            
            self.ud_screen_label.setStyleSheet("background: transparent; border: none;")
            self.ud_screen_label.raise_()
        # ---------------------------------------------

        # Back Button
        btn_back = QtWidgets.QPushButton()
        btn_back.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowLeft))
        btn_back.setIconSize(QtCore.QSize(35, 35))
        btn_back.setFixedSize(50, 50)
        btn_back.setFlat(True)
        btn_back.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        layout.addWidget(btn_back)

        layout.addStretch()

        # Login Form Card
        form_card = QtWidgets.QFrame()
        form_card.setFixedWidth(500)
        form_card.setStyleSheet("QFrame { background: white; border-radius: 20px; border: 1px solid #D1D1D1; }")
        form_layout = QtWidgets.QVBoxLayout(form_card)
        form_layout.setContentsMargins(40, 40, 40, 40)
        form_layout.setSpacing(15)

        title = QtWidgets.QLabel("Login")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #111; border: none;")
        form_layout.addWidget(title)

        self.edit_user = QtWidgets.QLineEdit()
        self.edit_user.setPlaceholderText("Username or Email")
        self.edit_user.setStyleSheet("padding: 15px; font-size: 16px; border: 1px solid #CCC; border-radius: 8px;")
        
        self.edit_pass = QtWidgets.QLineEdit()
        self.edit_pass.setPlaceholderText("Password")
        self.edit_pass.setEchoMode(QtWidgets.QLineEdit.Password)
        self.edit_pass.setStyleSheet("padding: 15px; font-size: 16px; border: 1px solid #CCC; border-radius: 8px;")

        btn_login = QtWidgets.QPushButton("LOG IN")
        btn_login.setStyleSheet("background: #111; color: white; padding: 15px; font-weight: bold; border-radius: 8px;")
        btn_login.clicked.connect(self.handle_manual_login)

        self.cb_remember_manual = QtWidgets.QCheckBox("Stay logged in")
        self.cb_remember_manual.setChecked(True)
        self.cb_remember_manual.setStyleSheet("color: #333; font-size: 14px; border: none;")
        
        # Sync them: If manual is clicked, QR is updated and vice versa
        self.cb_remember_manual.toggled.connect(self.cb_remember.setChecked)
        self.cb_remember.toggled.connect(self.cb_remember_manual.setChecked)
        
        form_layout.addWidget(self.edit_user)
        form_layout.addWidget(self.edit_pass)
        form_layout.addWidget(self.cb_remember_manual)
        form_layout.addWidget(btn_login)

        layout.addWidget(form_card, alignment=QtCore.Qt.AlignCenter)
        layout.addStretch()
        self.stack.addWidget(manual_widget)

    def handle_manual_login(self):
        user = self.edit_user.text()
        pw = self.edit_pass.text()
        uid = verify_user_credentials(user, pw)
        
        if uid:
            if self.cb_remember.isChecked(): # This will be the same state for both
                from database import save_session
                save_session(uid)
            self.user_id = uid
            self.timer.stop()
            self.accept()
        else:
            QtWidgets.QMessageBox.warning(self, "Login Failed", "Invalid username or password.")

    def check_login_status(self):
        if self.stack.currentIndex() == 0:
            uid = poll_for_login(self.session_id)
            if uid:
                # ADD THIS BLOCK BELOW
                if self.cb_remember.isChecked():
                    from database import save_session
                    save_session(uid)
                
                self.user_id = uid
                self.timer.stop()
                self.accept()

    def showEvent(self, event):
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self.showFullScreen)

def main():
    from database import init_db, get_active_session
    init_db()
    app = QtWidgets.QApplication(sys.argv)

    # 1. Check if we should skip login initially
    current_uid = get_active_session()

    while True:
        # If we don't have a session, show the Login screen
        if not current_uid:
            login = Login()
            if not login.exec_(): 
                break # User exited the login screen
            current_uid = login.user_id

        # 2. Show the Main Menu
        main_window = MainMenu(current_uid)
        main_window.showFullScreen()
        app.exec_()
        
        # 3. Check what happened after Main Menu closed
        if getattr(main_window, "logout_requested", False):
            # If they logged out, clear the local variable so the next loop shows Login()
            current_uid = None
            continue 
        else:
            # If they just closed the app normally, stop the loop
            break
            
    sys.exit()

if __name__ == "__main__":
    main()