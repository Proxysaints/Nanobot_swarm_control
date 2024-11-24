import time
import socket
import json
import logging
import psutil
import asyncio
import zlib
from datetime import datetime
from scapy.all import sniff  # Importing scapy for real Wi-Fi signal data capturing

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
    def __init__(self, signal_strength, rssi=None, rf_freq=None, signal_type="Wi-Fi"):
        self.signal_strength = signal_strength
        self.rssi = rssi  # RSSI value
        self.rf_freq = rf_freq  # RF frequency (real)
        self.signal_type = signal_type
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self):
        return {
            "signal_strength": self.signal_strength,
            "rssi": self.rssi,
            "rf_freq": self.rf_freq,
            "signal_type": self.signal_type,
            "timestamp": self.timestamp
        }

    def to_json(self):
        return json.dumps(self.to_dict())

# Get real signal information from Wi-Fi interfaces
def get_real_signal_strength():
    try:
        # Use system tools to gather signal information
        wireless_info = psutil.net_if_stats()
        if 'wlan0' in wireless_info:
            # Assuming we are capturing real-time data from a wireless interface
            signal_strength = psutil.net_if_stats()['wlan0'].speed  # Capture real-time speed or signal quality
            rssi = -70  # Placeholder for actual RSSI; this should be captured through network tools like `iwconfig` or `scapy`
            rf_freq = 2400  # Placeholder for actual RF frequency, should be gathered through scapy or `iwlist`
            return signal_strength, rssi, rf_freq
        else:
            logging.warning("Wi-Fi interface not found.")
            return 0.5, -70, 2400  # Default fallback values
    except Exception as e:
        logging.error(f"Error getting signal strength: {e}")
        return 0.5, -70, 2400  # Fallback to random values

# Sniffing Wi-Fi packets with Scapy to extract real signal information
def capture_signal():
    def packet_callback(packet):
        if packet.haslayer(Dot11):
            signal_strength = packet.dBm_AntSignal  # Extracting the dBm signal strength
            rf_freq = packet.Channel  # Frequency of the channel
            logging.info(f"Captured packet with signal strength: {signal_strength} dBm on channel: {rf_freq}")
            return signal_strength, rf_freq

    sniff(prn=packet_callback, count=1, iface="wlan0")  # Interface name can vary, check your system

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

class ConnectionPool:
    def __init__(self, host, port, pool_size=2):
        self.host = host
        self.port = port
        self.pool_size = pool_size
        self.pool = []
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize the pool with Wi-Fi modules."""
        for _ in range(self.pool_size):
            esp8266 = ESP8266(self.host, self.port)
            esp32 = ESP32(self.host, self.port)
            self.pool.append(esp8266)
            self.pool.append(esp32)

    async def get_connection(self):
        """Get a connection from the pool."""
        for connection in self.pool:
            if connection.reader is None and connection.writer is None:
                await connection.connect()
                return connection
        raise Exception("No available connections in the pool.")

    def return_connection(self, connection):
        """Return a connection to the pool."""
        self.pool.append(connection)

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

    # Create a connection pool with 2 connections (4 modules)
    pool = ConnectionPool(host, port, pool_size=2)

    try:
        while True:
            # Fetch real signal strength, RSSI, and RF frequency
            signal_strength, rssi, rf_freq = get_real_signal_strength()  # Get real values
            signal_data = Signal(signal_strength, rssi=rssi, rf_freq=rf_freq)

            # Get a connection from the pool
            connection = await pool.get_connection()

            # Send signals with retry mechanism
            await send_with_retry(connection, signal_data)

            # Receive signals (asynchronous)
            await connection.receive_signal()

            # Return the connection to the pool
            pool.return_connection(connection)

            await asyncio.sleep(2)

    except KeyboardInterrupt:
        logging.info("Communication interrupted by user.")

    finally:
        # Cleanup and close connections
        for connection in pool.pool:
            if connection.writer:
                connection.writer.close()
        logging.info("Connections closed.")

if __name__ == "__main__":
    asyncio.run(main())