import paho.mqtt.client as mqtt
import ssl

# --- NEW: HiveMQ Cloud Credentials ---
BROKER_ADDRESS = "e498150171fd4c8abc39c1d9f4e8c283.s1.eu.hivemq.cloud"
BROKER_PORT = 8883
USERNAME = "hivemq.webclient.1762679278869"
PASSWORD = "T491db0O#jtMLkm;?AR&"
# ---

class MqttClient:
    def __init__(self, client_id="ShrimpAppPython"):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        
        # --- NEW: Add authentication and TLS ---
        self.client.username_pw_set(USERNAME, PASSWORD)
        self.client.tls_set(tls_version=ssl.PROTOCOL_TLS_CLIENT)
        # ---
        
        self.client.on_connect = self.on_connect
        self.connected = False

    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print("MQTT: Python client connected successfully to HiveMQ.")
            self.connected = True
        else:
            print(f"MQTT: Python client failed to connect, code {rc}")

    def connect(self):
        try:
            # --- NEW: Connect to cloud port ---
            self.client.connect(BROKER_ADDRESS, BROKER_PORT)
            self.client.loop_start()  # Starts a background thread
        except Exception as e:
            print(f"MQTT: Could not connect to broker. Error: {e}")
            
    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
        print("MQTT: Python client disconnected.")

    def publish(self, topic, payload):
        if self.connected:
            self.client.publish(topic, payload)
        else:
            print(f"MQTT: Not connected. Cannot publish to {topic}")