import numpy as np
import concurrent.futures

# Particle Filter Class
class ParticleFilter:
    def __init__(self, num_particles=100, state_dim=3, noise_cov=0.1):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.particles = np.random.uniform(-1, 1, (self.num_particles, self.state_dim))
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.noise_cov = noise_cov

    def predict(self):
        """ Predict the next state of each particle by adding noise. """
        self.particles += np.random.normal(0, self.noise_cov, self.particles.shape)

    def update(self, measurement, sensor_noise):
        """ Update the particle weights based on the likelihood of each particle's position. """
        distances = np.linalg.norm(self.particles - measurement, axis=1)
        # The sensor noise increases with distance from the target
        sensor_noise_variation = sensor_noise * (1 + np.linalg.norm(measurement) / 10)  # more noise for far away
        self.weights = np.exp(-distances**2 / (2 * sensor_noise_variation**2))  # Vectorized computation
        self.weights /= np.sum(self.weights)  # Normalize the weights

    def resample(self):
        """ Resample particles based on their weights using systematic resampling. """
        cumulative_weights = np.cumsum(self.weights)
        random_values = np.random.uniform(0, 1, self.num_particles)
        indices = np.searchsorted(cumulative_weights, random_values)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate_position(self):
        """ Estimate the current position as the weighted average of the particles. """
        return np.average(self.particles, weights=self.weights, axis=0)

# Kalman Filter Class
class KalmanFilter:
    def __init__(self, state_dim=3, process_cov=0.1, measurement_cov=0.1):
        self.state_estimate = np.zeros(state_dim)
        self.state_cov = np.eye(state_dim) * process_cov
        self.process_cov = np.eye(state_dim) * process_cov
        self.measurement_cov = np.eye(state_dim) * measurement_cov

    def predict(self):
        """ Kalman filter prediction step (no control input). """
        self.state_cov += self.process_cov

    def update(self, measurement):
        """ Kalman update step based on measurement. """
        innovation = measurement - self.state_estimate
        innovation_cov = self.state_cov + self.measurement_cov
        kalman_gain = np.dot(self.state_cov, np.linalg.inv(innovation_cov))

        self.state_estimate += np.dot(kalman_gain, innovation)
        self.state_cov = np.dot(np.eye(self.state_cov.shape[0]) - kalman_gain, self.state_cov)

# PSO Class with vectorized operations and improved global command
class PSO:
    def __init__(self, num_particles=10, target_position=None, state_dim=3):
        self.num_particles = num_particles
        self.target_position = target_position
        self.state_dim = state_dim
        self.positions = np.random.uniform(-1, 1, (self.num_particles, self.state_dim))
        self.velocities = np.zeros_like(self.positions)
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.ones(self.num_particles) * float('inf')
        self.global_best_position = None
        self.global_best_score = float('inf')

    def update(self):
        inertia_weight = 0.5
        cognitive_weight = 1.5
        social_weight = 1.5

        # Calculate distances in parallel
        distances_to_target = np.linalg.norm(self.positions - self.target_position, axis=1)
        fitness_scores = distances_to_target

        # Update personal bests and global best in parallel
        self.personal_best_scores = np.minimum(self.personal_best_scores, fitness_scores)
        self.personal_best_positions[fitness_scores < self.personal_best_scores] = self.positions[fitness_scores < self.personal_best_scores]

        # Update global best position
        best_particle_idx = np.argmin(fitness_scores)
        if fitness_scores[best_particle_idx] < self.global_best_score:
            self.global_best_score = fitness_scores[best_particle_idx]
            self.global_best_position = self.positions[best_particle_idx]

        # Vectorized PSO velocity and position update
        inertia = inertia_weight * self.velocities
        cognitive = cognitive_weight * np.random.rand(self.num_particles, self.state_dim) * (self.personal_best_positions - self.positions)
        social = social_weight * np.random.rand(self.num_particles, self.state_dim) * (self.global_best_position - self.positions)

        self.velocities = inertia + cognitive + social
        self.positions += self.velocities

    def get_best_position(self):
        return self.global_best_position

# Nanobot Class with optimized position update
class Nanobot:
    def __init__(self, position):
        self.position = np.array(position)

    def update_position(self, global_command, local_command):
        """ Vectorized position update. """
        self.position += global_command * 0.5 + local_command * 0.5

# Combined PF, KF, and PSO for Nanobot Swarm Control with Multithreading
class SwarmControl:
    def __init__(self, num_particles=10, num_nanobots=5, target_position=None):
        self.num_particles = num_particles
        self.num_nanobots = num_nanobots
        self.pso = PSO(num_particles=num_particles, target_position=target_position)
        self.pf = ParticleFilter(num_particles=num_particles)
        self.kf = KalmanFilter()
        self.nanobots = [Nanobot(position=np.random.uniform(-1, 1, 3)) for _ in range(num_nanobots)]

    def update(self):
        # Update PSO for swarm optimization
        self.pso.update()

        # Parallelizing Particle Filter and Kalman Filter updates using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.update_nanobot_position, nanobot) for nanobot in self.nanobots]
            for future in futures:
                future.result()

        # Update each nanobot's position using global and local commands
        global_command = self.pso.get_best_position()
        local_commands = np.random.uniform(-1, 1, (self.num_nanobots, 3))  # Random local commands for each nanobot
        for i, nanobot in enumerate(self.nanobots):
            nanobot.update_position(global_command, local_commands[i])

    def update_nanobot_position(self, nanobot):
        pf_measurement = nanobot.position  # Use the current position as a measurement
        self.pf.predict()
        self.pf.update(pf_measurement, sensor_noise=0.1)
        self.pf.resample()

        self.kf.predict()
        self.kf.update(pf_measurement)

        # Combining PF and KF estimates
        estimated_position = (self.pf.estimate_position() + self.kf.state_estimate) / 2
        nanobot.position = estimated_position

    def get_nanobot_positions(self):
        return [nanobot.position for nanobot in self.nanobots]

# Main execution loop
if __name__ == "__main__":
    target_position = np.array([0, 0, 0])
    swarm_control = SwarmControl(num_particles=10, num_nanobots=5, target_position=target_position)

    # Main loop (simulate multiple updates)
    for _ in range(10):  # Update for 10 iterations
        swarm_control.update()  # Update nanobots' positions using PSO, PF, and KF
        print("Nanobot Positions:", swarm_control.get_nanobot_positions())