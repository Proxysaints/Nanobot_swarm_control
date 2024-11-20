import numpy as np

class ParticleFilter:
    def __init__(self, num_particles=100, state_dim=3, noise_cov=0.1):
        """
        Initialize a Particle Filter.
        """
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.particles = np.random.uniform(-1, 1, (self.num_particles, self.state_dim))
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.noise_cov = noise_cov
    
    def predict(self):
        """Predict the next state of each particle by adding noise."""
        noise = np.random.normal(0, self.noise_cov, self.particles.shape)
        self.particles += noise

    def update(self, measurement, sensor_noise):
        """
        Update the particle weights based on the likelihood of each particle's position.
        """
        distances = np.linalg.norm(self.particles - measurement, axis=1)  # Compute distances for all particles
        self.weights = np.exp(-distances**2 / (2 * sensor_noise**2))  # Vectorized computation for efficiency
        self.weights /= np.sum(self.weights)  # Normalize the weights

    def resample(self):
        """Resample particles based on their weights using the systematic resampling method."""
        cumulative_weights = np.cumsum(self.weights)  # Compute cumulative sum of the weights
        random_values = np.random.uniform(0, 1, self.num_particles)  # Random values for resampling
        indices = np.searchsorted(cumulative_weights, random_values)  # Efficient resampling
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles  # Reset the weights to uniform

    def estimate_position(self):
        """Estimate the current position as the weighted average of the particles."""
        return np.average(self.particles, weights=self.weights, axis=0)


class ParticleFilterWithKalman(ParticleFilter):
    def __init__(self, num_particles=100, state_dim=3, noise_cov=0.1, process_cov=0.1, measurement_cov=0.1):
        """
        Combine Particle Filter with Kalman Filter to improve state estimation.
        """
        super().__init__(num_particles, state_dim, noise_cov)
        # Kalman filter parameters
        self.state_estimate = np.zeros(state_dim)  # Initial state estimate
        self.state_cov = np.eye(state_dim) * process_cov  # Initial state covariance
        self.process_cov = np.eye(state_dim) * process_cov  # Process noise covariance
        self.measurement_cov = np.eye(state_dim) * measurement_cov  # Measurement noise covariance

    def predict(self):
        """
        Kalman prediction step.
        Update the state estimate using the process model (here we assume no control input).
        """
        self.state_estimate = self.state_estimate  # Predict state (no control model, just hold previous)
        self.state_cov = self.state_cov + self.process_cov  # Predict state covariance

        # Particle prediction step (Add noise to particles)
        noise = np.random.normal(0, self.noise_cov, self.particles.shape)
        self.particles += noise

    def update(self, measurement, sensor_noise):
        """
        Update step for both Kalman filter and particle filter.
        Kalman update uses the measurement to correct the state estimate.
        """
        # Kalman Filter Update
        innovation = measurement - self.state_estimate  # Measurement innovation
        innovation_cov = self.state_cov + self.measurement_cov  # Innovation covariance
        kalman_gain = np.dot(self.state_cov, np.linalg.inv(innovation_cov))  # Kalman gain
        
        # Correct the state estimate
        self.state_estimate = self.state_estimate + np.dot(kalman_gain, innovation)
        self.state_cov = np.dot(np.eye(self.state_dim) - kalman_gain, self.state_cov)

        # Particle filter update
        distances = np.linalg.norm(self.particles - measurement, axis=1)
        self.weights = np.exp(-distances**2 / (2 * sensor_noise**2))  # Vectorized computation for efficiency
        self.weights /= np.sum(self.weights)  # Normalize the weights

    def estimate_position(self):
        """Estimate the current position using both Kalman and particle estimates."""
        particle_estimate = np.average(self.particles, weights=self.weights, axis=0)
        # Combine Kalman and particle estimates by averaging them
        return (self.state_estimate + particle_estimate) / 2


# Example usage without simulated data:
if __name__ == "__main__":
    pf_kf = ParticleFilterWithKalman(num_particles=100, state_dim=3, process_cov=0.1, measurement_cov=0.1)

    # Replace this with actual measurement data as it's not simulated here
    measurement = np.array([0.5, 0.5, 0.5])  # Example measurement
    sensor_noise = 0.1  # Sensor noise

    for _ in range(10):  # Run for 10 iterations
        pf_kf.predict()  # Predict new state
        pf_kf.update(measurement, sensor_noise)  # Update with measurement
        estimated_position = pf_kf.estimate_position()  # Get combined estimate
        print(estimated_position)