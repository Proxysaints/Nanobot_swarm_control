import numpy as np

class HybridController:
    def __init__(self, num_nanos, target_position=None, field_source_position=None, ambient_temperature=25.0, num_particles=100):
        # Initialize basic parameters
        self.num_nanos = num_nanos
        self.target_position = target_position
        self.field_source_position = field_source_position
        self.ambient_temperature = ambient_temperature
        
        # Initialize Particle Filter (PF) and Kalman Filter (KF)
        self.pf = ParticleFilter(num_particles=num_particles)
        self.kf = KalmanFilter()
        
        # Initialize Particle Swarm Optimization (PSO) to optimize swarm movement
        self.pso = PSO(num_particles=num_particles, target_position=target_position)
        
        # Initialize any other necessary attributes
        self.lock = threading.Lock()

    def calculate_global_command(self, positions):
        """Calculate the global command based on the target position and swarm behavior."""
        if self.target_position is not None:
            # Calculate the centroid of the nanobots
            centroid = np.mean(positions, axis=0)
            # Calculate the global direction towards the target position
            direction_to_target = self.target_position - centroid
            return direction_to_target
        return np.zeros(3)

    def calculate_local_move_direction(self, nanobot, field_source_position):
        """Calculate the local move direction based on magnetic navigation and temperature."""
        # Use the magnetic gradient sensor to get the magnetic field direction
        magnetic_field, gradient = nanobot.sense_magnetic_field(field_source_position)
        
        # Apply temperature correction for magnetic field
        temp_factor = 1 - 0.01 * (self.ambient_temperature - 25)
        corrected_gradient = gradient * temp_factor
        
        # Return the local move direction influenced by the magnetic field gradient
        return corrected_gradient * 0.1  # Scaling factor for movement

    def update_target(self, nanobots_positions):
        """Update the target position based on swarm behavior (centroid) or predefined target."""
        if self.target_position is None:
            self.target_position = np.mean(nanobots_positions, axis=0)

    def update(self, nanobot_manager):
        """Update the nanobots' movement and sensor data."""
        nanobots_positions = nanobot_manager.get_positions()

        # Calculate the global command
        global_command = self.calculate_global_command(nanobots_positions)

        # Update the target if needed
        self.update_target(nanobots_positions)

        # Apply global command and magnetic navigation to each nanobot
        for nanobot in nanobot_manager.nanobots:
            # Use the Particle Filter (PF) to estimate the nanobot's position
            pf_measurement = nanobot_manager.position_system.update_position()
            self.pf.predict()
            self.pf.update(pf_measurement, sensor_noise=0.1, 
                           field_source_position=self.field_source_position, 
                           temperature=self.ambient_temperature)
            self.pf.resample()
            estimated_position = self.pf.estimate_position()
            nanobot.position = estimated_position

            # Apply local movement based on magnetic sensing and update temperature
            local_move_direction = self.calculate_local_move_direction(nanobot, self.field_source_position)
            nanobot.update_temperature()  # Update temperature from the sensor

            # Update the nanobot's position considering global command and local navigation
            nanobot.update_position(global_command, local_move_direction)

        # Use PSO to optimize the swarm's overall movement
        self.pso.update()

        # Update the 3D tracker with new nanobot positions
        nanobot_manager.tracker.update(nanobots_positions)

class ParticleFilter:
    def __init__(self, num_particles=100, state_dim=3, noise_cov=0.1, measurement_model=None):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.particles = np.random.uniform(-1, 1, (self.num_particles, self.state_dim))
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.noise_cov = noise_cov
        self.measurement_model = measurement_model if measurement_model else self.gaussian_likelihood
    
    def gaussian_likelihood(self, particle, measurement, sensor_noise):
        """Default Gaussian likelihood function."""
        distance = np.linalg.norm(particle - measurement)
        return np.exp(-distance**2 / (2 * sensor_noise**2))
    
    def predict(self):
        """Predict the next state of each particle by adding noise."""
        noise = np.random.normal(0, self.noise_cov, self.particles.shape)
        self.particles += noise

    def update(self, measurement, sensor_noise):
        """Update the particle weights based on the measurement."""
        self.weights = np.array([self.update_particle_weight(p, measurement, sensor_noise) for p in self.particles])
        total_weight = np.sum(self.weights)
        self.weights /= total_weight if total_weight > 0 else 1

    def update_particle_weight(self, particle, measurement, sensor_noise):
        """Update the particle's weight based on its likelihood of the measurement."""
        likelihood = self.measurement_model(particle, measurement, sensor_noise)
        return likelihood

    def resample(self):
        """Resample particles based on their weights using the systematic resampling method."""
        cumulative_weights = np.cumsum(self.weights)
        random_values = np.random.uniform(0, 1, self.num_particles)
        indices = np.searchsorted(cumulative_weights, random_values)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate_position(self):
        """Estimate the current position as the weighted average of the particles."""
        return np.average(self.particles, weights=self.weights, axis=0)


class KalmanFilter:
    def __init__(self, state_dim=3, process_cov=0.1, measurement_cov=0.1):
        self.state_estimate = np.zeros(state_dim)
        self.state_cov = np.eye(state_dim) * process_cov
        self.process_cov = np.eye(state_dim) * process_cov
        self.measurement_cov = np.eye(state_dim) * measurement_cov

    def predict(self):
        """Kalman prediction step."""
        self.state_estimate = self.state_estimate
        self.state_cov = self.state_cov + self.process_cov

    def update(self, measurement):
        """Update step for Kalman filter."""
        innovation = measurement - self.state_estimate
        innovation_cov = self.state_cov + self.measurement_cov
        kalman_gain = np.dot(self.state_cov, np.linalg.inv(innovation_cov))
        self.state_estimate += np.dot(kalman_gain, innovation)
        self.state_cov = np.dot(np.eye(self.state_cov.shape[0]) - kalman_gain, self.state_cov)


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