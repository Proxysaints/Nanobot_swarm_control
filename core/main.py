import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gps import SimulatedGpsPositioningSystem, RealGpsPositioningSystem
from sensors import SimulatedMagnetometer, SimulatedTemperatureSensor, RealMagnetometer, RealTemperatureSensor
from navigation import MagneticNanobot
from particle_filter import ParticleFilterWithMGS
from nanobot_manager import NanobotManagerWithMagneticSensing
from positioning.tracking_3d import Tracking3D
from controllers.hybrid_controller import HybridController
from filters.particle_filter import ParticleFilter
from filters.pso import PSO

# Toggle between simulation and real-world data
use_simulation = True  # Set this to False for real-world data

def setup_positioning_system():
    """Setup GPS system based on the use_simulation flag."""
    if use_simulation:
        return SimulatedGpsPositioningSystem()
    else:
        return RealGpsPositioningSystem()

def setup_sensor_system():
    """Setup sensor system based on the use_simulation flag."""
    if use_simulation:
        return SimulatedMagnetometer(), SimulatedTemperatureSensor()
    else:
        return RealMagnetometer(), RealTemperatureSensor()

if __name__ == "__main__":

    # Set up the position system based on simulation toggle
    position_system = setup_positioning_system()
    field_source_position = np.array([0, 0, 0])  # Position of magnetic field source (can be adjusted)

    # Initialize the 3D tracking system
    tracker = Tracking3D()

    # Define a target position for the nanobots
    target_position = np.array([0, 0, 0])  # Example target position for nanobots

    # Create the Hybrid Controller for managing nanobots towards target position
    hybrid_controller = HybridController(num_nanos=5, target_position=target_position)

    # Initialize the Nanobot Manager with Magnetic Sensing
    nanobot_manager = NanobotManagerWithMagneticSensing(position_system, tracker, field_source_position)

    # Add nanobots with sensor systems
    for i in range(5):
        # Create a Magnetic Nanobot with random initial positions
        nanobot = MagneticNanobot(position=[np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)])

        # Set the sensor system based on simulation toggle
        magnetometer, temperature_sensor = setup_sensor_system()
        nanobot.magnetometer = magnetometer
        nanobot.temperature_sensor = temperature_sensor

        # Add nanobot to the manager
        nanobot_manager.add_nanobot(nanobot)

    # Set up the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Main loop for controlling nanobots (simulation or real control)
    while True:
        # Update the positioning system (e.g., update GPS or simulated position)
        position_system.update_position()

        # Use hybrid controller to update nanobots' positions
        hybrid_controller.update(nanobot_manager, hybrid_controller)

        # Get the positions of all nanobots
        positions = nanobot_manager.get_positions()

        # Clear the current plot
        ax.cla()

        # Plot the positions of nanobots
        x = [pos[0] for pos in positions]
        y = [pos[1] for pos in positions]
        z = [pos[2] for pos in positions]
        ax.scatter(x, y, z, c='r', marker='o')

        # Optionally, plot the target position (can be visualized as a green dot)
        ax.scatter(target_position[0], target_position[1], target_position[2], c='g', marker='x')

        # Redraw the plot
        plt.draw()

        # Pause to update the plot in real-time
        plt.pause(0.1)  # Adjust this for visualization speed

        # Sleep for a while before the next update (simulation step)
        time.sleep(1)