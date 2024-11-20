import serial
import time
import numpy as np
import random

# Wireless Communication Implementation
class WirelessCommunication:
    def __init__(self, transmission_range=10.0, signal_loss_factor=0.1):
        """
        Initialize the wireless communication system with parameters for range and signal loss.
        :param transmission_range: Maximum range for signal transmission
        :param signal_loss_factor: Factor controlling the rate of signal attenuation over distance
        """
        self.transmission_range = transmission_range
        self.signal_loss_factor = signal_loss_factor  # Simulate signal degradation
    
    def send_signal(self, sender_position, receiver_position, signal_strength):
        """
        Simulate sending a wireless signal from a sender to a receiver.
        The signal's strength is attenuated based on distance and the signal loss factor.
        :param sender_position: The position of the sender device.
        :param receiver_position: The position of the receiver device.
        :param signal_strength: The initial strength of the signal.
        :return: The received signal strength at the receiver.
        """
        distance = np.linalg.norm(np.array(sender_position) - np.array(receiver_position))
        if distance <= self.transmission_range:
            # Signal attenuation based on distance and signal loss factor
            received_signal_strength = signal_strength / (1 + self.signal_loss_factor * distance)
            return received_signal_strength
        return 0  # No signal if out of range
    
    def receive_signal(self, signal_strength):
        """
        Simulate a device receiving a signal. If the signal strength is below a threshold, 
        the signal is not considered received.
        :param signal_strength: The strength of the received signal.
        :return: True if the signal is successfully received, False if it is lost or corrupted.
        """
        noise = random.uniform(0, 0.5)  # Simulate random noise
        if signal_strength > noise:
            return True  # Signal received successfully
        return False  # Signal lost or corrupted by noise

# UART Communication Implementation
class SignalTransmitter:
    def __init__(self, port="/dev/ttyS0", baudrate=9600):
        self.serial_port = serial.Serial(port, baudrate, timeout=1)
        print(f"Transmitter initialized on {port} with baudrate {baudrate}")

    def send_data(self, data):
        try:
            print(f"Transmitting data: {data}")
            self.serial_port.write(data.encode())  # Send data as bytes
        except Exception as e:
            print(f"Signal Transmitter Error: {e}")

    def receive_data(self):
        try:
            data = self.serial_port.readline().decode().strip()  # Read a line of data
            if data:
                print(f"Received data: {data}")
                return data
            else:
                return None
        except Exception as e:
            print(f"Signal Receiver Error: {e}")
            return None

# Main Logic for Combined Transmitter and Wireless Communication
if __name__ == "__main__":
    # Initialize UART communication system
    transmitter_uart = SignalTransmitter(port="/dev/ttyS0", baudrate=9600)
    receiver_uart = SignalTransmitter(port="/dev/ttyS1", baudrate=9600)  # Receiver port can differ
    
    # Initialize wireless communication system
    wireless_comm = WirelessCommunication(transmission_range=15.0, signal_loss_factor=0.2)
    
    # Define positions for transmitter and receiver in a 3D space
    transmitter_position = [0, 0, 0]
    receiver_position = [5, 5, 5]
    
    # Test communication loop
    try:
        while True:
            # Simulate sending data over UART
            message = "Hello from Transmitter!"
            transmitter_uart.send_data(message)
            
            # Simulate wireless signal transmission with attenuation
            signal_strength = 1.0  # Initial signal strength
            received_signal_strength = wireless_comm.send_signal(transmitter_position, receiver_position, signal_strength)
            if wireless_comm.receive_signal(received_signal_strength):
                print(f"Wireless signal successfully received at {receiver_position}")
            else:
                print("Wireless signal lost due to attenuation or noise.")
            
            # Check if receiver UART has any incoming data
            received_message = receiver_uart.receive_data()
            if received_message:
                print(f"Message received via UART: {received_message}")
            
            # Wait 2 seconds before sending the next message
            time.sleep(2)

    except KeyboardInterrupt:
        print("Communication interrupted by user.")
    finally:
        transmitter_uart.serial_port.close()
        receiver_uart.serial_port.close()
        print("Serial ports closed.")