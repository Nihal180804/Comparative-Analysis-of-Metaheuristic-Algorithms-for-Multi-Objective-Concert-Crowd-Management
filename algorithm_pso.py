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
    try:
        pbest_fitnesses = env.calculate_batch_fitness(positions)
    except Exception:
        pbest_fitnesses = np.array([env.calculate_solution_fitness(pos) for pos in positions])

    # Initialize gBest (global best)
    gbest_index = np.argmin(pbest_fitnesses)
    gbest_position = pbest_positions[gbest_index].copy()
    gbest_fitness = pbest_fitnesses[gbest_index]

    # 2. PSO Loop
    for iteration in range(max_iterations):
        # generate random factors per-iteration (broadcasted)
        r1 = np.random.rand()
        r2 = np.random.rand()

        # Vectorized velocity update
        v_inertia = w * velocities
        v_cognitive = c1 * r1 * (pbest_positions - positions)
        v_social = c2 * r2 * (gbest_position[None, :] - positions)

        velocities = v_inertia + v_cognitive + v_social

        # Update positions (continuous -> discrete)
        positions = np.clip(positions + velocities, 0, env.num_gates - 1)
        positions = np.round(positions).astype(int)

        # Batch evaluate all particle fitnesses
        try:
            current_fitnesses = env.calculate_batch_fitness(positions)
        except Exception:
            current_fitnesses = np.array([env.calculate_solution_fitness(pos) for pos in positions])

        # Update pBest where improved
        improved_mask = current_fitnesses < pbest_fitnesses
        if np.any(improved_mask):
            pbest_positions[improved_mask] = positions[improved_mask].copy()
            pbest_fitnesses[improved_mask] = current_fitnesses[improved_mask]

            # Update gBest
            new_gbest_idx = int(np.argmin(pbest_fitnesses))
            if pbest_fitnesses[new_gbest_idx] < gbest_fitness:
                gbest_fitness = pbest_fitnesses[new_gbest_idx]
                gbest_position = pbest_positions[new_gbest_idx].copy()

        # Early stopping for visual feedback (optional)
        if iteration % (max_iterations // 10) == 0:
            print(f"  PSO Iter {iteration}/{max_iterations}: Global Best Fitness = {gbest_fitness:.2f}")

    return gbest_position
