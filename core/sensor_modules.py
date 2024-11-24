import numpy as np
import time
from w1thermsensor import W1ThermSensor
import smbus
from particle_filter import ParticleFilter  # Assuming you have the ParticleFilter class defined elsewhere

# ------------------- Magnetic Gradient Sensor -------------------

class MagneticGradientSensor:
    def __init__(self, position, temperature=25.0):
        self.position = np.array(position)
        self.temperature = temperature

    def read_magnetic_field(self, field_source_position, field_strength=1.0):
        distance = np.linalg.norm(self.position - field_source_position)
        temp_factor = 1 - 0.01 * (self.temperature - 25)
        magnetic_field = (field_strength * temp_factor) / (distance**2)
        gradient = (self.position - field_source_position) / distance
        return magnetic_field, gradient

    def update_temperature(self, ambient_temperature):
        self.temperature = ambient_temperature


class Magnetometer:
    def __init__(self, bus=1, address=0x1E):
        self.bus = smbus.SMBus(bus)
        self.address = address

    def read_magnetic_field(self):
        try:
            data = self.bus.read_i2c_block_data(self.address, 3, 6)
            x = np.int16((data[0] << 8) | data[1])  # Convert to signed 16-bit
            y = np.int16((data[2] << 8) | data[3])
            z = np.int16((data[4] << 8) | data[5])
            return np.array([x, y, z])
        except Exception as e:
            print(f"Magnetometer Error: {e}")
            return np.array([0, 0, 0])  # Default to zero vector


class SimulatedMagnetometer:
    def read_magnetic_field(self):
        # Generate a random magnetic field vector
        return np.random.normal(0, 1, size=3)


# ------------------- Temperature Sensor -------------------

class TemperatureSensor:
    def __init__(self):
        self.sensor = W1ThermSensor()

    def get_temperature(self):
        return self.sensor.get_temperature()


class SimulatedTemperatureSensor:
    def get_temperature(self):
        # Simulate ambient temperature changes
        return 25.0 + np.random.normal(0, 0.5)


# ------------------- Particle Filter with Magnetic Gradient Sensor and Temperature Effect -------------------

class ParticleFilterWithMGS(ParticleFilter):
    def update(self, measurement, sensor_noise, field_source_position, temperature):
        for i, particle in enumerate(self.particles):
            position_distance = np.linalg.norm(particle - measurement)
            position_likelihood = np.exp(-position_distance**2 / (2 * sensor_noise**2))
            magnetic_field, gradient = nanobot.sense_magnetic_field(field_source_position)
            temp_factor = 1 - 0.01 * (temperature - 25)
            magnetic_likelihood = np.exp(-np.linalg.norm(gradient)**2 * temp_factor / (2 * sensor_noise**2))
            likelihood = position_likelihood * magnetic_likelihood
            self.weights[i] = likelihood
        self.weights /= np.sum(self.weights)


# ------------------- Nanobot Class -------------------

class MagneticNanobot:
    def __init__(self, position=None):
        self.position = np.array(position if position is not None else [0, 0, 0])  # Initialize position
        self.sensor = MagneticGradientSensor(position=self.position)
        self.magnetometer = Magnetometer()  # Default to real magnetometer
        self.temperature_sensor = TemperatureSensor()  # Default to real temperature sensor
        self.sensors = {"magnetic": self.sensor}

    def sense_magnetic_field(self, field_source_position):
        magnetic_field = self.magnetometer.read_magnetic_field()
        gradient = (self.position - field_source_position) / np.linalg.norm(self.position - field_source_position)
        return magnetic_field, gradient

    def update_position(self, global_command, local_move_direction):
        self.position = self.position + local_move_direction
        self.position = np.clip(self.position, -1, 1)

    def apply_magnetic_navigation(self, field_source_position):
        magnetic_field, gradient = self.sense_magnetic_field(field_source_position)
        move_direction = gradient * 0.1
        return move_direction

    def update_temperature(self):
        self.temperature = self.temperature_sensor.get_temperature()


# ------------------- Nanobot Manager with Magnetic Sensing -------------------

class NanobotManagerWithMagneticSensing:
    def __init__(self, tracker, field_source_position=None, ambient_temperature=25.0):
        self.nanobots = []
        self.tracker = tracker
        self.field_source_position = field_source_position
        self.pf = ParticleFilterWithMGS(num_particles=100)
        self.ambient_temperature = ambient_temperature

    def add_nanobot(self, nanobot):
        self.nanobots.append(nanobot)

    def get_positions(self):
        return [nanobot.position for nanobot in self.nanobots]

    def update(self, controller):
        nanobots_positions = self.get_positions()
        controller.update_target(nanobots_positions)
        global_command = controller.calculate_global_command(nanobots_positions)

        # Update Particle Filter (PF)
        for nanobot in self.nanobots:
            # Using simulated position as a measurement instead of GPS data
            pf_measurement = nanobot.position
            self.pf.predict()
            self.pf.update(pf_measurement, sensor_noise=0.1, field_source_position=self.field_source_position, temperature=self.ambient_temperature)
            self.pf.resample()
            estimated_position = self.pf.estimate_position()
            nanobot.position = estimated_position

        # Update Nanobots
        for nanobot in self.nanobots:
            nanobot.update_temperature()  # Read the temperature
            magnetic_navigation_direction = nanobot.apply_magnetic_navigation(self.field_source_position)
            nanobot.update_position(global_command, magnetic_navigation_direction)

        self.tracker.update(nanobots_positions)


# ------------------- Main Execution Loop -------------------

if __name__ == "__main__":
    use_simulation = True  # Toggle this for real or simulated mode

    tracker = None  # Replace with actual tracker (e.g., Tracking3D) if needed
    field_source_position = np.array([0, 0, 0])

    nanobot_manager = NanobotManagerWithMagneticSensing(tracker, field_source_position)

    # Add nanobots
    for _ in range(5):
        nanobot = MagneticNanobot()
        nanobot.magnetometer = SimulatedMagnetometer()  # Replace with simulated magnetometer
        nanobot.temperature_sensor = SimulatedTemperatureSensor()  # Replace with simulated sensor
        nanobot_manager.add_nanobot(nanobot)

    while True:
        nanobot_manager.update(None)  # Replace `None` with a valid controller if available
        print("Nanobot Positions:", nanobot_manager.get_positions())
        time.sleep(1)