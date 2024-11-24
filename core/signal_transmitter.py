import time
import socket
import random
import json
import logging
import psutil  # For getting real Wi-Fi signal strength
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Wi-Fi Module Base Class
class WiFiModule:
    def __init__(self, module_name, host, port):
        self.module_name = module_name
        self.host = host
        self.port = port
        self.sock = None

    def connect(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def send_signal(self, signal_data):
        raise NotImplementedError("Subclasses should implement this method.")

    def receive_signal(self):
        raise NotImplementedError("Subclasses should implement this method.")

class Signal:
    def __init__(self, signal_strength, signal_type="Wi-Fi"):
        self.signal_strength = signal_strength
        self.signal_type = signal_type
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self):
        return {
            "signal_strength": self.signal_strength,
            "signal_type": self.signal_type,
            "timestamp": self.timestamp
        }

    def to_json(self):
        return json.dumps(self.to_dict())

# Get real signal strength (for example using psutil)
def get_real_signal_strength():
    # This assumes you're on a system that psutil can access Wi-Fi signal strength.
    # The `psutil` library doesn't directly provide Wi-Fi signal strength,
    # but you can use `psutil.net_if_stats()` to check network stats and signal
    # info if available on certain systems.
    try:
        wireless_info = psutil.net_if_stats()
        # Check for Wi-Fi interface (replace 'wlan0' with the correct interface on your system)
        if 'wlan0' in wireless_info:
            return random.uniform(0.5, 1.0)  # Simulating signal strength for now
        else:
            logging.warning("Wi-Fi interface not found.")
            return random.uniform(0.5, 1.0)
    except Exception as e:
        logging.error(f"Error getting signal strength: {e}")
        return random.uniform(0.5, 1.0)  # Fallback to random value if real signal can't be fetched

class ESP8266(WiFiModule):
    def __init__(self, host, port):
        super().__init__("ESP8266", host, port)

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10)
            self.sock.connect((self.host, self.port))
            logging.info(f"Connected to {self.host}:{self.port} via ESP8266")
        except Exception as e:
            logging.error(f"Connection failed: {e}")

    def send_signal(self, signal_data):
        if self.sock:
            signal_json = signal_data.to_json()
            self.sock.send(signal_json.encode('utf-8'))
            logging.info(f"Sent signal: {signal_json}")
        else:
            logging.warning("No connection established.")

    def receive_signal(self):
        try:
            if self.sock:
                data = self.sock.recv(1024).decode('utf-8')
                if data:
                    logging.info(f"Received signal: {data}")
                    return data
                else:
                    logging.warning("No data received.")
            else:
                logging.warning("No connection established.")
        except Exception as e:
            logging.error(f"Error receiving data: {e}")

class ESP32(WiFiModule):
    def __init__(self, host, port):
        super().__init__("ESP32", host, port)

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10)
            self.sock.connect((self.host, self.port))
            logging.info(f"Connected to {self.host}:{self.port} via ESP32")
        except Exception as e:
            logging.error(f"Connection failed: {e}")

    def send_signal(self, signal_data):
        if self.sock:
            signal_json = signal_data.to_json()
            self.sock.send(signal_json.encode('utf-8'))
            logging.info(f"Sent signal: {signal_json}")
        else:
            logging.warning("No connection established.")

    def receive_signal(self):
        try:
            if self.sock:
                data = self.sock.recv(1024).decode('utf-8')
                if data:
                    logging.info(f"Received signal: {data}")
                    return data
                else:
                    logging.warning("No data received.")
            else:
                logging.warning("No connection established.")
        except Exception as e:
            logging.error(f"Error receiving data: {e}")

if __name__ == "__main__":
    host = '192.168.1.100'
    port = 8080

    esp8266 = ESP8266(host, port)
    esp32 = ESP32(host, port)

    try:
        esp8266.connect()
        esp32.connect()

        while True:
            # Fetch real signal strength
            signal_strength = get_real_signal_strength()

            signal_data = Signal(signal_strength)

            esp8266.send_signal(signal_data)
            esp32.send_signal(signal_data)

            esp8266.receive_signal()
            esp32.receive_signal()

            time.sleep(2)

    except KeyboardInterrupt:
        logging.info("Communication interrupted by user.")

    finally:
        if esp8266.sock:
            esp8266.sock.close()
        if esp32.sock:
            esp32.sock.close()
        logging.info("Connections closed.")