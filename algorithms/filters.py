import numpy as np
from multiprocessing import Pool

class ParticleFilter:
    def __init__(self, num_particles=100, state_dim=3, noise_cov=0.1, measurement_model=None):
        """
        Initialize a Particle Filter.
        measurement_model: A callable that defines how to compute the likelihood of a measurement.
        """
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.particles = np.random.uniform(-1, 1, (self.num_particles, self.state_dim))
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.noise_cov = noise_cov
        
        # Default measurement model (Gaussian likelihood)
        self.measurement_model = measurement_model if measurement_model else self.gaussian_likelihood
    
    def gaussian_likelihood(self, particle, measurement, sensor_noise):
        """Default Gaussian likelihood function."""
        distance = np.linalg.norm(particle - measurement)
        return np.exp(-distance**2 / (2 * sensor_noise**2))
    
    def predict(self):
        """Predict the next state of each particle by adding noise."""
        noise = np.random.normal(0, self.noise_cov, self.particles.shape)
        self.particles += noise

    def update_particle_weight(self, particle, measurement, sensor_noise):
        """
        Update the particle's weight based on its likelihood of the measurement.
        This function allows flexibility in defining different likelihood functions.
        """
        likelihood = self.measurement_model(particle, measurement, sensor_noise)
        return likelihood

    def update(self, measurement, sensor_noise):
        """Update the particle weights based on the measurement."""
        # Parallelize the update process
        with Pool() as pool:
            self.weights = pool.map(lambda p: self.update_particle_weight(p, measurement, sensor_noise), self.particles)
        
        # Normalize the weights
        total_weight = np.sum(self.weights)
        self.weights /= total_weight if total_weight > 0 else 1

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


class ParticleFilterWithKalman(ParticleFilter):
    def __init__(self, num_particles=100, state_dim=3, noise_cov=0.1, process_cov=0.1, measurement_cov=0.1, measurement_model=None):
        """
        Combine Particle Filter with Kalman Filter to improve state estimation.
        """
        super().__init__(num_particles, state_dim, noise_cov, measurement_model)
        self.state_estimate = np.zeros(state_dim)
        self.state_cov = np.eye(state_dim) * process_cov
        self.process_cov = np.eye(state_dim) * process_cov
        self.measurement_cov = np.eye(state_dim) * measurement_cov

    def predict(self):
        """Kalman prediction step."""
        self.state_estimate = self.state_estimate
        self.state_cov = self.state_cov + self.process_cov
        noise = np.random.normal(0, self.noise_cov, self.particles.shape)
        self.particles += noise

    def update(self, measurement, sensor_noise):
        """Update step for Kalman filter and Particle filter."""
        innovation = measurement - self.state_estimate
        innovation_cov = self.state_cov + self.measurement_cov
        kalman_gain = np.dot(self.state_cov, np.linalg.inv(innovation_cov))
        self.state_estimate += np.dot(kalman_gain, innovation)
        self.state_cov = np.dot(np.eye(self.state_dim) - kalman_gain, self.state_cov)
        
        super().update(measurement, sensor_noise)  # Update particles with the new measurement

    def estimate_position(self):
        """Estimate position using both Kalman and Particle estimates."""
        particle_estimate = np.average(self.particles, weights=self.weights, axis=0)
        return (self.state_estimate + particle_estimate) / 2


# Example usage with a custom non-Gaussian likelihood function:
if __name__ == "__main__":
    def custom_likelihood(particle, measurement, sensor_noise):
        """Example custom likelihood (Poisson distribution-based)."""
        distance = np.linalg.norm(particle - measurement)
        return np.exp(-distance / sensor_noise)  # A simple non-Gaussian likelihood

    pf_kf = ParticleFilterWithKalman(num_particles=100, state_dim=3, process_cov=0.1, measurement_cov=0.1, measurement_model=custom_likelihood)

    measurement = np.array([0.5, 0.5, 0.5])
    sensor_noise = 0.1

    for _ in range(10):
        pf_kf.predict()
        pf_kf.update(measurement, sensor_noise)
        estimated_position = pf_kf.estimate_position()
        print(estimated_position)