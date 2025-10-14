import numpy as np
from crowd_management_env import CrowdManagementEnv

def solve_pso(env: CrowdManagementEnv, num_particles=30, max_iterations=100, w=0.8, c1=2.0, c2=2.0) -> np.ndarray:
    """
    Implements the Particle Swarm Optimization (PSO) algorithm for gate assignment.
    """
    np.random.seed(42)
    num_attendees = env.num_attendees
    
    # 1. Initialize Particles
    # Positions (solutions) are discrete (gate indices), but movement is continuous
    positions = np.array([env.get_random_solution() for _ in range(num_particles)])
    velocities = np.zeros_like(positions, dtype=float)
    
    # Initialize pBest (personal best)
    pbest_positions = positions.copy()
    pbest_fitnesses = np.array([env.calculate_solution_fitness(pos) for pos in positions])
    
    # Initialize gBest (global best)
    gbest_index = np.argmin(pbest_fitnesses)
    gbest_position = pbest_positions[gbest_index].copy()
    gbest_fitness = pbest_fitnesses[gbest_index]

    # 2. PSO Loop
    for iteration in range(max_iterations):
        r1, r2 = np.random.rand(2)
        
        for i in range(num_particles):
            # 3. Update Velocity (In a continuous space)
            v_inertia = w * velocities[i]
            v_cognitive = c1 * r1 * (pbest_positions[i] - positions[i]) # Towards pBest
            v_social = c2 * r2 * (gbest_position - positions[i])        # Towards gBest
            
            velocities[i] = v_inertia + v_cognitive + v_social

            # 4. Update Position (Quantized for the discrete problem)
            # The position is updated continuously but must be quantized to a valid gate index
            positions[i] = np.clip(positions[i] + velocities[i], 0, env.num_gates - 1).round().astype(int)
            
            # 5. Evaluate and Update pBest
            current_fitness = env.calculate_solution_fitness(positions[i])
            if current_fitness < pbest_fitnesses[i]:
                pbest_fitnesses[i] = current_fitness
                pbest_positions[i] = positions[i].copy()
                
                # 6. Update gBest
                if current_fitness < gbest_fitness:
                    gbest_fitness = current_fitness
                    gbest_position = positions[i].copy()

        # Early stopping for visual feedback (optional)
        if iteration % (max_iterations // 10) == 0:
            print(f"  PSO Iter {iteration}/{max_iterations}: Global Best Fitness = {gbest_fitness:.2f}")

    return gbest_position
