Add Signal Transmission: The magnetic field changes are transmitted wirelessly through an external interface (e.g., using radio frequency (RF) signals).


import numpy as np
import time
import serial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from brian2 import *

# Assuming RFID, RSSI, Kalman Filter, Spiking Neural Network (SNN), and other required classes are already defined.

# Particle Filter Implementation
class ParticleFilter:
    def __init__(self, num_particles=100, state_dim=3, noise_cov=0.1):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.particles = np.random.uniform(-1, 1, (self.num_particles, self.state_dim))  # Random initial positions
        self.weights = np.ones(self.num_particles) / self.num_particles  # Equal weight for all particles initially
        self.noise_cov = noise_cov
    
    def predict(self):
        # Predict the next state for all particles with some noise
        noise = np.random.normal(0, self.noise_cov, self.particles.shape)
        self.particles += noise

    def update(self, measurement, sensor_noise):
        # Update weights based on how close the measurement is to the particle
        for i, particle in enumerate(self.particles):
            # Calculate the likelihood of the particle's state based on the measurement
            distance = np.linalg.norm(particle - measurement)
            likelihood = np.exp(-distance**2 / (2 * sensor_noise**2))  # Gaussian likelihood
            self.weights[i] = likelihood
        
        # Normalize weights
        self.weights /= np.sum(self.weights)

    def resample(self):
        # Resample particles based on their weights
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles  # Reset weights to equal

    def estimate_position(self):
        # Estimate the position as the weighted mean of the particles
        return np.average(self.particles, weights=self.weights, axis=0)

# Particle Swarm Optimization (PSO) Implementation
class PSO:
    def __init__(self, num_particles=10, target_position=None):
        self.num_particles = num_particles
        self.target_position = target_position
        self.positions = np.random.uniform(-1, 1, (self.num_particles, 3))  # Initialize positions
        self.velocities = np.zeros_like(self.positions)  # Initial velocities
        self.personal_best_positions = self.positions.copy()  # Best known positions for each particle
        self.personal_best_scores = np.ones(self.num_particles) * float('inf')  # Initialize with large scores
        self.global_best_position = None
        self.global_best_score = float('inf')

    def update(self):
        # Update each particle’s velocity and position
        inertia_weight = 0.5
        cognitive_weight = 1.5
        social_weight = 1.5

        for i in range(self.num_particles):
            # Compute fitness (distance to target)
            distance_to_target = np.linalg.norm(self.positions[i] - self.target_position)
            fitness_score = distance_to_target
            
            # Update personal best
            if fitness_score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = fitness_score
                self.personal_best_positions[i] = self.positions[i]
            
            # Update global best
            if fitness_score < self.global_best_score:
                self.global_best_score = fitness_score
                self.global_best_position = self.positions[i]
            
            # Update velocity (PSO equation)
            inertia = inertia_weight * self.velocities[i]
            cognitive = cognitive_weight * np.random.rand() * (self.personal_best_positions[i] - self.positions[i])
            social = social_weight * np.random.rand() * (self.global_best_position - self.positions[i])
            
            self.velocities[i] = inertia + cognitive + social
            
            # Update position
            self.positions[i] += self.velocities[i]

    def get_best_position(self):
        return self.global_best_position

# Nanobot Manager with PF and PSO
class NanobotManagerWithPFandPSO:
    def __init__(self, position_system, tracker, pso_target_position=None):
        self.nanobots = []
        self.position_system = position_system
        self.tracker = tracker
        self.pso = PSO(num_particles=10, target_position=pso_target_position)
        self.pf = ParticleFilter(num_particles=100)

    def add_nanobot(self, nanobot):
        self.nanobots.append(nanobot)

    def get_positions(self):
        return [nanobot.position for nanobot in self.nanobots]

    def update(self, controller):
        nanobots_positions = self.get_positions()
        controller.update_target(nanobots_positions)
        global_command = controller.calculate_global_command(nanobots_positions)

        # Update PSO for swarm optimization
        self.pso.update()

        # Update Particle Filter for position estimation
        for nanobot in self.nanobots:
            pf_measurement = np.random.normal(nanobot.position, 0.1)  # Simulated noisy position
            self.pf.predict()
            self.pf.update(pf_measurement, sensor_noise=0.1)
            self.pf.resample()
            estimated_position = self.pf.estimate_position()
            nanobot.position = estimated_position  # Update nanobot position with PF estimate

        # Update each nanobot's position
        for nanobot in self.nanobots:
            rfid_tag = nanobot.sensors["rfid"].read_tag()
            local_move_direction = nanobot.sense_environment(rfid_tag, self.position_system)
            nanobot.update_position(global_command, local_move_direction)

        self.tracker.update(nanobots_positions)

# Hybrid Controller Class
class HybridController:
    def __init__(self, num_nanos, target_position=None):
        self.num_nanos = num_nanos
        self.target_position = target_position
        self.centralized_controller = CentralizedController()  # Centralized controller
        self.decentralized_controllers = [DecentralizedController() for _ in range(num_nanos)]  # Decentralized controllers

    def calculate_global_command(self, positions):
        """
        Calculate a global command for all nanobots based on centralized control.
        This could involve calculating a target position or formation.
        """
        if self.target_position:
            return self.target_position  # Direct the swarm to the target position
        # Add more complex logic for centralized control if needed.
        return np.mean(positions, axis=0)  # Example: move towards the centroid of all nanobots

    def decentralized_adjustment(self, nanobot, neighbor_positions):
        """
        Each nanobot will adjust its behavior based on local sensors and proximity to neighbors.
        This could include obstacle avoidance, collision detection, or other local decisions.
        """
        # Example: adjust speed or direction based on proximity to other nanobots
        local_command = nanobot.position  # Default to current position
        for neighbor in neighbor_positions:
            distance = np.linalg.norm(nanobot.position - neighbor.position)
            if distance < 0.5:  # If too close, move away
                local_command = nanobot.position + (nanobot.position - neighbor.position) * 0.1
        return local_command

    def update(self, nanobot_manager, controller):
        """
        Update the nanobots' positions using both centralized and decentralized control.
        """
        # Get the positions of all nanobots
        positions = nanobot_manager.get_positions()

        # Centralized control: Calculate global command for all nanobots
        global_command = self.calculate_global_command(positions)

        # Update each nanobot using both centralized and decentralized control
        for i, nanobot in enumerate(nanobot_manager.nanobots):
            # Get positions of neighboring nanobots
            neighbor_positions = [nb.position for j, nb in enumerate(nanobot_manager.nanobots) if i != j]

            # First, apply decentralized control (local behavior)
            local_move_direction = self.decentralized_adjustment(nanobot, neighbor_positions)

            # Now apply centralized control (global command)
            nanobot.update_position(global_command, local_move_direction)

        # Optionally, update the tracker or any additional systems here
        nanobot_manager.tracker.update(positions)

# Main Execution Loop with Hybrid Controller
if __name__ == "__main__":
    position_system = RealGpsPositioningSystem()
    tracker = Tracking3D()

    # Create the Hybrid Controller with 5 nanobots and a target position
    hybrid_controller = HybridController(num_nanos=5, target_position=np.array([0, 0, 0]))

    nanobot_manager = NanobotManagerWithPFandPSO(position_system, tracker)

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





































