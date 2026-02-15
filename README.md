# ShrimpSense: Deep Learning-Based Biomass Calculation for Post-Larval Shrimp

**ShrimpSense** is a specialized computer vision and IoT system designed for precision aquaculture. It utilizes deep learning to automate the counting of post-larval shrimp and calculates real-time biomass to provide optimized feeding recommendations.

---

## 👨‍🔬 Research Team
* **Mervin James Batuhan**
* **Paul Isaiah Cachin**
* **Kazuki Ogata**
* **Ery Jay Pisalbon**
* **Aaron Jonathan Valencia**

**Project Title:** Design of a Deep Learning-Based Biomass Calculation for Post-Larval Shrimp Stocking with Feed Optimization System

---

## 📖 Operational Guide: How to Use

### 1. Authentication
The system supports two secure methods for accessing the control interface:
* **QR Handshake:** Upon launching, the machine generates a unique Session ID and displays it as a QR code. Open the **ShrimpSense** mobile app, scan the code, and the machine will automatically log you in using your cloud credentials.
* **Manual Login:** If the machine is offline, tap "Try another way" to access the manual login screen. Enter your registered email/username and password. The system verifies these against a local encrypted cache for offline reliability.

### 2. Biomass Calculation Process
1.  **Set Target:** Tap "SET TARGET" to enter the specific number of shrimp you intend to stock.
2.  **Start System:** Tap "START" to begin the camera feed and deep learning inference.
3.  **Automated Counting:** As shrimp pass through the sensor area, the YOLO-based detector identifies them and the centroid tracker counts them only once as they cross the detection line.
4.  **Target Reached:** Once the count hits your target, the system sends an MQTT command to close the intake door/servo automatically.

### 3. Feed Optimization & Dispensing
* **Real-time Metrics:** The system continuously calculates the total biomass (weight) and the required feed dosage based on the current count.
* **Dispense:** Tap "DISPENSE FEED" to trigger the IoT-connected feeder to release the exact calculated amount of nutrients.

### 4. Data Management
* **Local Save:** Tap "SAVE" to store the session data (count, weight, time) into the local SQLite database.
* **Cloud Sync:** Access the History window to view past sessions and tap "Sync to Cloud" to upload local records to the MongoDB Atlas dashboard for long-term tracking.

---

## 📊 Technical Implementation



### Deep Learning Inference
The core of the system is a **YOLO (You Only Look Once)** model exported to **ONNX** format. This allows the Raspberry Pi 5 to perform high-speed detection without needing a dedicated GPU by utilizing the ONNX Runtime CPU Execution Provider.

### IoT & Control Logic
* **MQTT (HiveMQ):** Used as the communication backbone for low-latency hardware triggers (Servos/Pumps).
* **Centroid Tracking:** Assigns unique IDs to detected objects to prevent double-counting as shrimp move across the frame.

---
**© 2025-2026 | Computer Engineering | Project Design 1 & 2 | Team 16**
