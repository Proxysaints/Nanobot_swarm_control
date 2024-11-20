import numpy as np
import concurrent.futures
import threading

# Particle Filter Class (Thread-safe)
class ParticleFilter:
    def __init__(self, num_particles=100, state_dim=3, noise_cov=0.1):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.particles = np.random.uniform(-1, 1, (self.num_particles, self.state_dim))
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.noise_cov = noise_cov
        self.lock = threading.Lock()  # Lock for thread safety

    def predict(self):
        with self.lock:
            self.particles += np.random.normal(0, self.noise_cov, self.particles.shape)

    def update(self, measurement, sensor_noise):
        with self.lock:
            distances = np.linalg.norm(self.particles - measurement, axis=1)
            self.weights = np.exp(-distances**2 / (2 * sensor_noise**2))  # Vectorized computation
            self.weights /= np.sum(self.weights)  # Normalize the weights

    def resample(self):
        with self.lock:
            cumulative_weights = np.cumsum(self.weights)
            random_values = np.random.uniform(0, 1, self.num_particles)
            indices = np.searchsorted(cumulative_weights, random_values)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate_position(self):
        with self.lock:
            return np.average(self.particles, weights=self.weights, axis=0)

# Kalman Filter Class (Thread-safe)
class KalmanFilter:
    def __init__(self, state_dim=3, process_cov=0.1, measurement_cov=0.1):
        self.state_estimate = np.zeros(state_dim)
        self.state_cov = np.eye(state_dim) * process_cov
        self.process_cov = np.eye(state_dim) * process_cov
        self.measurement_cov = np.eye(state_dim) * measurement_cov
        self.lock = threading.Lock()

    def predict(self):
        with self.lock:
            self.state_cov += self.process_cov

    def update(self, measurement):
        with self.lock:
            innovation = measurement - self.state_estimate
            innovation_cov = self.state_cov + self.measurement_cov
            kalman_gain = np.dot(self.state_cov, np.linalg.inv(innovation_cov))

            self.state_estimate += np.dot(kalman_gain, innovation)
            self.state_cov = np.dot(np.eye(self.state_cov.shape[0]) - kalman_gain, self.state_cov)

# Particle Swarm Optimization (PSO) Class (Optimized)
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

        distances_to_target = np.linalg.norm(self.positions - self.target_position, axis=1)
        fitness_scores = distances_to_target

        self.personal_best_scores = np.minimum(self.personal_best_scores, fitness_scores)
        self.personal_best_positions[fitness_scores < self.personal_best_scores] = self.positions[fitness_scores < self.personal_best_scores]

        best_particle_idx = np.argmin(fitness_scores)
        if fitness_scores[best_particle_idx] < self.global_best_score:
            self.global_best_score = fitness_scores[best_particle_idx]
            self.global_best_position = self.positions[best_particle_idx]

        inertia = inertia_weight * self.velocities
        cognitive = cognitive_weight * np.random.rand(self.num_particles, self.state_dim) * (self.personal_best_positions - self.positions)
        social = social_weight * np.random.rand(self.num_particles, self.state_dim) * (self.global_best_position - self.positions)

        self.velocities = inertia + cognitive + social
        self.positions += self.velocities

    def get_best_position(self):
        return self.global_best_position

# Nanobot Class with Thread Safety and Enhanced Update Logic
class Nanobot:
    def __init__(self, position):
        self.position = np.array(position)
        self.lock = threading.Lock()

    def update_position(self, global_command, local_command):
        with self.lock:
            self.position += global_command * 0.5 + local_command * 0.5

# SwarmControl Class (Enhanced Thread Safety, Error Handling, Adaptive Noise Handling)
class SwarmControl:
    def __init__(self, num_particles=10, num_nanobots=5, initial_target_position=None):
        self.num_particles = num_particles
        self.num_nanobots = num_nanobots
        self.target_position = np.array(initial_target_position)
        self.pso = PSO(num_particles=num_particles, target_position=self.target_position)
        self.pf = ParticleFilter(num_particles=num_particles)
        self.kf = KalmanFilter()
        self.nanobots = [Nanobot(position=np.random.uniform(-1, 1, 3)) for _ in range(num_nanobots)]
        self.lock = threading.Lock()

    def update_target(self, step):
        try:
            self.target_position += np.sin(0.1 * step) * np.random.uniform(-0.1, 0.1, self.target_position.shape)
            self.pso.target_position = self.target_position
        except Exception as e:
            print(f"Error in updating target position: {e}")

    def adaptive_sensor_noise(self, nanobot_position):
        distance_to_target = np.linalg.norm(nanobot_position - self.target_position)
        # More complex noise model based on velocity and proximity
        noise = max(0.1, 1.0 / (1 + distance_to_target)) + 0.05 * np.random.rand()
        return noise

    def update_nanobot_position(self, nanobot):
        try:
            global_command = self.pso.get_best_position()
            local_command = np.random.uniform(-0.1, 0.1, nanobot.position.shape)
            sensor_noise = self.adaptive_sensor_noise(nanobot.position)

            # Update PF and KF
            self.pf.update(nanobot.position, sensor_noise)
            self.kf.update(nanobot.position)

            # Use PF for position estimation
            estimated_position = self.pf.estimate_position()

            nanobot.update_position(global_command, local_command)
            nanobot.position = estimated_position  # Use PF's estimate
        except Exception as e:
            print(f"Error in updating nanobot position: {e}")

    def update(self, step):
        try:
            self.update_target(step)
            self.pso.update()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.update_nanobot_position, nanobot) for nanobot in self.nanobots]
                concurrent.futures.wait(futures)

        except Exception as e:
            print(f"An error occurred in swarm control update: {e}")