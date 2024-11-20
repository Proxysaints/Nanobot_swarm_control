Signal Acquisition: Magnetic Gradients & Magnetothermal effects

Import numpy as np
import time
import serial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from brian2 import *

# Magnetic Gradient Sensor Class with Temperature Effect
class MagneticGradientSensor:
    def __init__(self, position, temperature=25.0):
        self.position = np.array(position)  # Position of the sensor
        self.temperature = temperature  # Initial temperature in degrees Celsius
    
    def read_magnetic_field(self, field_source_position, field_strength=1.0):
        """
        Simulate reading the magnetic field at the sensor's position
        based on the distance to a field source and the temperature effect.
        """
        distance = np.linalg.norm(self.position - field_source_position)
        
        # Temperature effect on magnetic field strength
        temp_factor = 1 - 0.01 * (self.temperature - 25)  # Simplified temperature effect
        magnetic_field = (field_strength * temp_factor) / (distance**2)  # Inverse-square law with temperature effect
        gradient = (self.position - field_source_position) / distance  # Directional gradient
        
        return magnetic_field, gradient
    
    def update_temperature(self, ambient_temperature):
        """
        Simulate temperature changes over time, possibly influenced by nearby objects or ambient conditions.
        """
        # Simple model for temperature change
        self.temperature = ambient_temperature

# Nanobot Class with Magnetic Gradient Sensor and Magnetothermal Effect
class MagneticNanobot:
    def __init__(self, position):
        self.position = np.array(position)
        self.sensor = MagneticGradientSensor(position)  # Initialize MGS with temperature
        self.sensors = {"magnetic": self.sensor}
    
    def sense_magnetic_field(self, field_source_position):
        """
        Get the magnetic field and gradient from the sensor, accounting for temperature effect.
        """
        magnetic_field, gradient = self.sensors["magnetic"].read_magnetic_field(field_source_position)
        return magnetic_field, gradient
    
    def update_position(self, global_command, local_move_direction):
        """
        Update the nanobot's position based on external control commands
        and local sensor readings (magnetic gradient).
        """
        # Move based on global command and local sensor feedback
        self.position = self.position + local_move_direction
        self.position = np.clip(self.position, -1, 1)  # Constrain position to within bounds
    
    def apply_magnetic_navigation(self, field_source_position):
        """
        Use the magnetic gradient data to navigate the nanobot.
        Move toward or away from the magnetic field source, considering temperature.
        """
        _, gradient = self.sense_magnetic_field(field_source_position)
        # Adjust movement based on magnetic gradient and temperature (if necessary)
        move_direction = gradient * 0.1  # Scale movement by gradient strength
        return move_direction

# Particle Filter with Magnetic Gradient Sensor and Temperature Effect
class ParticleFilterWithMGS(ParticleFilter):
    def update(self, measurement, sensor_noise, field_source_position, temperature):
        """
        Update the particle filter using both position measurements and magnetic gradient,
        considering the magnetothermal effect.
        """
        for i, particle in enumerate(self.particles):
            # Position measurement likelihood (based on noisy position)
            position_distance = np.linalg.norm(particle - measurement)
            position_likelihood = np.exp(-position_distance**2 / (2 * sensor_noise**2))
            
            # Magnetic gradient likelihood (using MGS data, adjusted for temperature)
            magnetic_field, gradient = nanobot.sense_magnetic_field(field_source_position)
            # Temperature effect on magnetic likelihood
            temp_factor = 1 - 0.01 * (temperature - 25)  # Simplified temperature adjustment
            magnetic_likelihood = np.exp(-np.linalg.norm(gradient)**2 * temp_factor / (2 * sensor_noise**2))
            
            # Combine position and magnetic likelihoods
            likelihood = position_likelihood * magnetic_likelihood
            self.weights[i] = likelihood
        
        # Normalize weights
        self.weights /= np.sum(self.weights)

# Nanobot Manager with Magnetic Gradient Sensing and Temperature Updates
class NanobotManagerWithMagneticSensing:
    def __init__(self, position_system, tracker, field_source_position=None, ambient_temperature=25.0):
        self.nanobots = []
        self.position_system = position_system
        self.tracker = tracker
        self.field_source_position = field_source_position  # Magnetic field source
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

        # Update Particle Filter for position estimation, considering temperature
        for nanobot in self.nanobots:
            pf_measurement = np.random.normal(nanobot.position, 0.1)  # Simulated noisy position
            self.pf.predict()
            self.pf.update(pf_measurement, sensor_noise=0.1, field_source_position=self.field_source_position, temperature=self.ambient_temperature)
            self.pf.resample()
            estimated_position = self.pf.estimate_position()
            nanobot.position = estimated_position  # Update nanobot position with PF estimate

        # Update each nanobot's position and temperature
        for nanobot in self.nanobots:
            # Update temperature (simulate temperature change)
            nanobot.sensor.update_temperature(self.ambient_temperature)

            # Use magnetic gradient for navigation, considering temperature
            magnetic_navigation_direction = nanobot.apply_magnetic_navigation(self.field_source_position)
            nanobot.update_position(global_command, magnetic_navigation_direction)

        self.tracker.update(nanobots_positions)

# Main Execution Loop with Magnetothermal Effect
if __name__ == "__main__":
    position_system = RealGpsPositioningSystem()
    tracker = Tracking3D()

    # Define the position of the magnetic field source (e.g., a magnet)
    field_source_position = np.array([0, 0, 0])  # Place the magnetic source at origin

    # Define ambient temperature
    ambient_temperature = 30.0  # Example: 30 degrees Celsius

    nanobot_manager = NanobotManagerWithMagneticSensing(position_system, tracker, field_source_position, ambient_temperature)

    # Add nanobots to the manager
    for i in range(5):
        nanobot = MagneticNanobot(position=[np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
        nanobot_manager.add_nanobot(nanobot)

    # Main loop
    while True:
        # Update positioning system and nanobots
        position_system.update_position()
        
        # Use hybrid control to update nanobots
        hybrid_controller.update(nanobot_manager, hybrid_controller)
        
        # Output nanobot positions
        print("Nanobot Positions:", nanobot_manager.get_positions())
        
        # Sleep for a while before the next update
        time.sleep(1)