import numpy as np

class ParticleFilter:
    def __init__(self, num_particles=100, state_dim=3, noise_cov=0.1):
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
        The likelihood is based on how close the particle is to the actual measurement.
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


class ParticleFilterWithMGS(ParticleFilter):
    def update(self, measurement, sensor_noise, field_source_position, temperature, nanobot):
        """
        Update the particle filter using both position measurements and magnetic gradient,
        considering the magnetothermal effect.
        """
        # Calculate the position likelihoods (distance between particles and measurement)
        distances = np.linalg.norm(self.particles - measurement, axis=1)
        position_likelihoods = np.exp(-distances**2 / (2 * sensor_noise**2))
        
        # Calculate magnetic gradient likelihoods for all particles
        magnetic_likelihoods = np.zeros(self.num_particles)
        for i, particle in enumerate(self.particles):
            # Get the magnetic field and gradient for each particle
            magnetic_field, gradient = nanobot.sense_magnetic_field(field_source_position)
            
            # Temperature adjustment (more accurate temperature model)
            temp_factor = 1 - 0.01 * (temperature - 25)  # Simplified linear adjustment
            gradient_norm = np.linalg.norm(gradient)
            
            # Combine magnetic gradient likelihood with the temperature adjustment
            magnetic_likelihoods[i] = np.exp(-gradient_norm**2 * temp_factor / (2 * sensor_noise**2))
        
        # Combine the position and magnetic likelihoods
        combined_likelihoods = position_likelihoods * magnetic_likelihoods
        
        # Update particle weights
        self.weights = combined_likelihoods
        self.weights /= np.sum(self.weights)  # Normalize the weights