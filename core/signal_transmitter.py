import time
import socket
import random
import json
import logging
import psutil
import asyncio
import zlib
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Wi-Fi Module Base Class
class WiFiModule:
    def __init__(self, module_name, host, port):
        self.module_name = module_name
        self.host = host
        self.port = port
        self.reader = None
        self.writer = None

    async def connect(self):
        """Asynchronous connection method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement this method.")

    async def send_signal(self, signal_data):
        """Asynchronously send signal data to the server."""
        raise NotImplementedError("Subclasses should implement this method.")

    async def receive_signal(self):
        """Asynchronously receive signal data from the server."""
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
    try:
        wireless_info = psutil.net_if_stats()
        if 'wlan0' in wireless_info:
            return random.uniform(0.5, 1.0)  # Simulating signal strength for now
        else:
            logging.warning("Wi-Fi interface not found.")
            return random.uniform(0.5, 1.0)
    except Exception as e:
        logging.error(f"Error getting signal strength: {e}")
        return random.uniform(0.5, 1.0)  # Fallback to random value

class AsyncWiFiModule(WiFiModule):
    async def connect(self):
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        logging.info(f"Connected to {self.host}:{self.port} via {self.module_name}")

    async def send_signal(self, signal_data):
        signal_json = signal_data.to_json()
        compressed_signal = compress_signal(signal_json)  # Compress the signal data before sending
        self.writer.write(compressed_signal)
        await self.writer.drain()
        logging.info(f"Sent signal: {signal_json}")

    async def receive_signal(self):
        try:
            data = await self.reader.read(1024)
            if data:
                decompressed_data = decompress_signal(data)
                logging.info(f"Received signal: {decompressed_data}")
        except Exception as e:
            logging.error(f"Error receiving data: {e}")

class ESP8266(AsyncWiFiModule):
    def __init__(self, host, port):
        super().__init__("ESP8266", host, port)

class ESP32(AsyncWiFiModule):
    def __init__(self, host, port):
        super().__init__("ESP32", host, port)

async def send_with_retry(module, signal_data, retries=3):
    for attempt in range(retries):
        try:
            await module.send_signal(signal_data)
            return True
        except Exception as e:
            logging.error(f"Send attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(2 ** attempt)
    logging.error("All retries failed.")
    return False

def compress_signal(data):
    """Compress the signal data."""
    return zlib.compress(data.encode('utf-8'))

def decompress_signal(data):
    """Decompress the signal data."""
    return zlib.decompress(data).decode('utf-8')

async def main():
    host = '192.168.1.100'
    port = 8080

    esp8266 = ESP8266(host, port)
    esp32 = ESP32(host, port)

    await esp8266.connect()
    await esp32.connect()

    try:
        while True:
            # Fetch real signal strength
            signal_strength = get_real_signal_strength()
            signal_data = Signal(signal_strength)

            # Send signals with retry mechanism
            await send_with_retry(esp8266, signal_data)
            await send_with_retry(esp32, signal_data)

            # Receive signals (asynchronous)
            await esp8266.receive_signal()
            await esp32.receive_signal()

            await asyncio.sleep(2)

    except KeyboardInterrupt:
        logging.info("Communication interrupted by user.")

    finally:
        # Cleanup and close connections
        if esp8266.writer:
            esp8266.writer.close()
        if esp32.writer:
            esp32.writer.close()
        logging.info("Connections closed.")

if __name__ == "__main__":
    asyncio.run(main())