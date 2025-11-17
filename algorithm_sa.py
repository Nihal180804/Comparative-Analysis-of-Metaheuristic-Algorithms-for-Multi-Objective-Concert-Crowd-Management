import numpy as np
from crowd_management_env import CrowdManagementEnv

def solve_sa(env: CrowdManagementEnv, max_iterations=2000, initial_temp=1000.0, cooling_rate=0.995) -> np.ndarray:
    """
    Implements the Simulated Annealing (SA) algorithm for gate assignment.
    """
    np.random.seed(42)
    num_attendees = env.num_attendees
    
    # 1. Initialize
    current_solution = env.get_random_solution()
    current_fitness = env.calculate_solution_fitness(current_solution)
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    temperature = initial_temp

    # 2. SA Loop
    for iteration in range(max_iterations):
        # Generate a neighbor solution with a mix of small and large moves
        neighbor_solution = current_solution.copy()
        r = np.random.rand()
        if r < 0.6:
            # Small move: change one attendee's gate (ensure different gate chosen)
            attendee_to_change = np.random.randint(0, num_attendees)
            new_gate = np.random.randint(0, env.num_gates)
            # avoid trivial move
            if env.num_gates > 1:
                while new_gate == neighbor_solution[attendee_to_change]:
                    new_gate = np.random.randint(0, env.num_gates)
            neighbor_solution[attendee_to_change] = new_gate
        elif r < 0.9:
            # Medium move: swap assignments between two attendees
            a, b = np.random.choice(num_attendees, size=2, replace=False)
            neighbor_solution[a], neighbor_solution[b] = neighbor_solution[b], neighbor_solution[a]
        else:
            # Large move: perturb several random attendees
            num_changes = max(1, num_attendees // 20)
            for _ in range(num_changes):
                idx = np.random.randint(0, num_attendees)
                neighbor_solution[idx] = np.random.randint(0, env.num_gates)
        
        neighbor_fitness = env.calculate_solution_fitness(neighbor_solution)
        
        # Calculate Energy/Fitness Difference
        delta_fitness = neighbor_fitness - current_fitness

        # Acceptance Criteria
        if delta_fitness < 0:
            # Accept better solution
            current_solution = neighbor_solution
            current_fitness = neighbor_fitness
        elif temperature > 0:
            # Accept worse solution with probability exp(-delta_fitness / T)
            acceptance_probability = np.exp(-delta_fitness / temperature)
            if np.random.rand() < acceptance_probability:
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness

        # Update Best Solution Found
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_solution = current_solution.copy()

        # Cool Down the Temperature (slow cooling to allow exploration)
        temperature *= cooling_rate
        
        # Early stopping for visual feedback (optional)
        if iteration % max(1, (max_iterations // 10)) == 0:
            print(f"  SA Iter {iteration}/{max_iterations}: Best Fitness = {best_fitness:.2f}, Temp = {temperature:.2f}")


    return best_solution
