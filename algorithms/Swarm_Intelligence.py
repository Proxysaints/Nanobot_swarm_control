import numpy as np
import concurrent.futures

# Particle Filter Class (unchanged)
class ParticleFilter:
    def __init__(self, num_particles=100, state_dim=3, noise_cov=0.1):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.particles = np.random.uniform(-1, 1, (self.num_particles, self.state_dim))
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.noise_cov = noise_cov

    def predict(self):
        self.particles += np.random.normal(0, self.noise_cov, self.particles.shape)

    def update(self, measurement, sensor_noise):
        distances = np.linalg.norm(self.particles - measurement, axis=1)
        self.weights = np.exp(-distances**2 / (2 * sensor_noise**2))
        self.weights /= np.sum(self.weights)

    def resample(self):
        cumulative_weights = np.cumsum(self.weights)
        random_values = np.random.uniform(0, 1, self.num_particles)
        indices = np.searchsorted(cumulative_weights, random_values)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate_position(self):
        return np.average(self.particles, weights=self.weights, axis=0)

# Swarm Communication Class
class SwarmCommunication:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.positions = np.zeros((num_agents, 3))  # Use numpy array for positions
        self.velocities = np.zeros((num_agents, 3))  # Use numpy array for velocities

    def update_position(self, agent_id, position):
        self.positions[agent_id] = position

    def update_velocity(self, agent_id, velocity):
        self.velocities[agent_id] = velocity

    def get_neighborhood_positions(self, agent_id, radius=1.0):
        agent_pos = self.positions[agent_id]
        distances = np.linalg.norm(self.positions - agent_pos, axis=1)
        neighbors = self.positions[distances <= radius]
        return neighbors

# Nanobot Class
class Nanobot:
    def __init__(self, position, agent_id, swarm_comm):
        self.position = np.array(position)
        self.velocity = np.zeros_like(position)
        self.agent_id = agent_id
        self.swarm_comm = swarm_comm

    def communicate(self):
        self.swarm_comm.update_position(self.agent_id, self.position)
        self.swarm_comm.update_velocity(self.agent_id, self.velocity)

    def update_position(self, global_command, local_command):
        # Update position with the weighted global and local commands
        self.velocity = global_command * 0.7 + local_command * 0.3
        self.position += self.velocity
        self.communicate()

# SwarmControl Class
class SwarmControl:
    def __init__(self, num_nanobots=5, initial_target_position=None):
        self.num_nanobots = num_nanobots
        self.target_position = np.array(initial_target_position)
        self.pf = ParticleFilter(num_particles=100)
        self.swarm_comm = SwarmCommunication(num_nanobots)
        self.nanobots = [
            Nanobot(position=np.random.uniform(-1, 1, 3), agent_id=i, swarm_comm=self.swarm_comm)
            for i in range(num_nanobots)
        ]

    def group_decision_making(self):
        # Calculate the collective center of the swarm
        swarm_center = np.mean(self.swarm_comm.positions, axis=0)
        # Adjust the target position based on swarm dynamics
        self.target_position = (self.target_position + swarm_center) / 2

    def update_nanobot_positions(self):
        # Vectorized batch operations instead of using a thread per nanobot
        global_commands = self.target_position - self.swarm_comm.positions  # Move toward target
        local_commands = np.zeros_like(global_commands)
        
        # Calculate local commands by averaging neighbors' positions
        for i in range(self.num_nanobots):
            local_neighbors = self.swarm_comm.get_neighborhood_positions(i)
            if len(local_neighbors) > 0:
                local_commands[i] = np.mean(local_neighbors, axis=0) - self.swarm_comm.positions[i]
        
        # Apply velocity adjustments (combining global and local commands)
        self.swarm_comm.velocities = global_commands * 0.7 + local_commands * 0.3
        self.swarm_comm.positions += self.swarm_comm.velocities  # Update positions
        # Update communication after batch operations
        for i in range(self.num_nanobots):
            self.nanobots[i].communicate()

    def update(self, step):
        self.group_decision_making()
        self.update_nanobot_positions()

# Simulation Execution
if __name__ == "__main__":
    num_nanobots = 10
    initial_target_position = [5.0, 5.0, 5.0]
    swarm_control = SwarmControl(num_nanobots=num_nanobots, initial_target_position=initial_target_position)

    for step in range(100):
        swarm_control.update(step)
        print(f"Step {step}: Target Position: {swarm_control.target_position}")