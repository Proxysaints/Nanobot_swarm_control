import time
import socket
import random
import json
from datetime import datetime

# Wi-Fi Module Base Class
class WiFiModule:
    def __init__(self, module_name, host, port):
        """
        Initialize the Wi-Fi module with its unique module name and communication settings.
        :param module_name: Name of the Wi-Fi module (e.g., ESP8266, ESP32)
        :param host: Host address (IP) of the device to communicate with
        :param port: Port for communication
        """
        self.module_name = module_name
        self.host = host
        self.port = port
        self.sock = None

    def connect(self):
        """
        Establish a socket connection to the given host and port.
        :return: socket object
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def send_signal(self, signal_data):
        """
        Send a signal over the network using the Wi-Fi module.
        :param signal_data: The signal data (as a Signal object or dictionary)
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def receive_signal(self):
        """
        Receive a signal over the network.
        :return: The signal data received from the network
        """
        raise NotImplementedError("Subclasses should implement this method.")

# Signal Data Format Class
class Signal:
    def __init__(self, signal_strength, signal_type="Wi-Fi"):
        """
        Create a signal with a given strength and type.
        :param signal_strength: The strength of the signal (float)
        :param signal_type: The type of signal (e.g., Wi-Fi, Bluetooth)
        """
        self.signal_strength = signal_strength
        self.signal_type = signal_type
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self):
        """
        Convert the signal to a dictionary format for easy transmission.
        :return: Dictionary representation of the signal
        """
        return {
            "signal_strength": self.signal_strength,
            "signal_type": self.signal_type,
            "timestamp": self.timestamp
        }

    def to_json(self):
        """
        Convert the signal to a JSON string for easy transmission.
        :return: JSON string representation of the signal
        """
        return json.dumps(self.to_dict())

# Example ESP8266 Wi-Fi Module
class ESP8266(WiFiModule):
    def __init__(self, host, port):
        super().__init__("ESP8266", host, port)

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10)  # Set a timeout for connection
            self.sock.connect((self.host, self.port))
            print(f"Connected to {self.host}:{self.port} via ESP8266")
        except socket.timeout:
            print(f"Connection to {self.host}:{self.port} timed out.")
        except socket.error as e:
            print(f"Connection failed: {e}")
        except Exception as e:
            print(f"Unexpected error during connection: {e}")

    def send_signal(self, signal_data):
        try:
            if self.sock:
                signal_json = signal_data.to_json()  # Convert signal to JSON
                self.sock.send(signal_json.encode('utf-8'))
                print(f"Sent signal: {signal_json}")
            else:
                print("No connection established to send signal.")
        except socket.error as e:
            print(f"Failed to send signal: {e}")
        except Exception as e:
            print(f"Unexpected error during signal transmission: {e}")

    def receive_signal(self):
        try:
            if self.sock:
                data = self.sock.recv(1024).decode('utf-8')
                print(f"Received signal: {data}")
                return data
            else:
                print("No connection established to receive signal.")
        except socket.error as e:
            print(f"Failed to receive signal: {e}")
        except Exception as e:
            print(f"Unexpected error during signal reception: {e}")

# Example ESP32 Wi-Fi Module
class ESP32(WiFiModule):
    def __init__(self, host, port):
        super().__init__("ESP32", host, port)

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10)  # Set a timeout for connection
            self.sock.connect((self.host, self.port))
            print(f"Connected to {self.host}:{self.port} via ESP32")
        except socket.timeout:
            print(f"Connection to {self.host}:{self.port} timed out.")
        except socket.error as e:
            print(f"Connection failed: {e}")
        except Exception as e:
            print(f"Unexpected error during connection: {e}")

    def send_signal(self, signal_data):
        try:
            if self.sock:
                signal_json = signal_data.to_json()  # Convert signal to JSON
                self.sock.send(signal_json.encode('utf-8'))
                print(f"Sent signal: {signal_json}")
            else:
                print("No connection established to send signal.")
        except socket.error as e:
            print(f"Failed to send signal: {e}")
        except Exception as e:
            print(f"Unexpected error during signal transmission: {e}")

    def receive_signal(self):
        try:
            if self.sock:
                data = self.sock.recv(1024).decode('utf-8')
                print(f"Received signal: {data}")
                return data
            else:
                print("No connection established to receive signal.")
        except socket.error as e:
            print(f"Failed to receive signal: {e}")
        except Exception as e:
            print(f"Unexpected error during signal reception: {e}")

# Main Logic for Wireless Communication
if __name__ == "__main__":
    # Define server host and port for communication
    host = '192.168.1.100'  # Example IP address of the receiver device
    port = 8080  # Example port number

    # Initialize Wi-Fi modules
    esp8266 = ESP8266(host, port)
    esp32 = ESP32(host, port)

    # Test communication loop
    try:
        # Connect to the receiver
        esp8266.connect()
        esp32.connect()

        while True:
            # Simulate sending signal with different signal strengths
            signal_strength = random.uniform(0.5, 1.0)
            print(f"Sending signal with strength: {signal_strength}")

            # Create a Signal object
            signal_data = Signal(signal_strength)

            # Send signal from ESP8266 and ESP32
            esp8266.send_signal(signal_data)
            esp32.send_signal(signal_data)

            # Receive signals (simulate reception)
            esp8266.receive_signal()
            esp32.receive_signal()

            # Wait before next transmission
            time.sleep(2)

    except KeyboardInterrupt:
        print("Communication interrupted by user.")

    finally:
        # Close the connections if open
        if esp8266.sock:
            esp8266.sock.close()
        if esp32.sock:
            esp32.sock.close()
        print("Connections closed.")