import numpy as np
from crowd_management_env import CrowdManagementEnv

# Helper function to convert gate assignments to nodes in the ACO graph (for visualization only)
# In this implementation, the "graph" is conceptual: Att_i -> Gate_j
# The "edges" are the (i, j) combinations, and pheromones are on (attendee, gate) pairings.

def solve_aco(env: CrowdManagementEnv, num_ants=10, max_iterations=50, rho=0.1, alpha=1.0, beta=5.0) -> np.ndarray:
    """
    Implements the Ant Colony Optimization (ACO) algorithm for gate assignment.
    
    Pheromones are stored on the assignment: Pheromone[attendee_i, gate_j]
    """
    np.random.seed(42)
    num_attendees = env.num_attendees
    num_gates = env.num_gates
    
    # Pheromone matrix: dimensions (num_attendees, num_gates)
    # Pheromones represent the desirability of assigning attendee 'i' to gate 'j'
    pheromones = np.ones((num_attendees, num_gates)) / num_gates
    
    # Heuristic matrix (Visibility): inverse of distance
    # Lower distance (higher visibility) is better
    distance_matrix = env.get_distance_matrix()
    
    # Avoid division by zero for identical locations; add a small epsilon
    # The heuristic encourages assignments to closer gates
    heuristic = 1.0 / (distance_matrix + 1e-6) 
    
    best_solution = None
    best_fitness = float('inf')

    # 1. ACO Loop
    for iteration in range(max_iterations):
        
        all_ant_solutions = []
        all_ant_fitnesses = []
        
        # 2. Ant Construction Phase
        for ant in range(num_ants):
            ant_solution = np.zeros(num_attendees, dtype=int)
            
            # Each ant "builds" a solution (assigns gates to all attendees)
            for attendee_idx in range(num_attendees):
                
                # Calculate selection probability for each gate
                pheromone_contrib = pheromones[attendee_idx] ** alpha
                heuristic_contrib = heuristic[attendee_idx] ** beta
                
                # Combined probability numerator
                prob_numerator = pheromone_contrib * heuristic_contrib
                
                # Total probability denominator
                probabilities = prob_numerator / np.sum(prob_numerator)
                
                # Select a gate based on probability
                assigned_gate = np.random.choice(num_gates, p=probabilities)
                ant_solution[attendee_idx] = assigned_gate
                
            # Evaluate the completed solution
            fitness = env.calculate_solution_fitness(ant_solution)
            
            all_ant_solutions.append(ant_solution)
            all_ant_fitnesses.append(fitness)
            
            # Update global best solution
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = ant_solution.copy()

        # 3. Pheromone Evaporation
        pheromones *= (1 - rho)
        
        # 4. Pheromone Reinforcement (Deposit)
        # We reinforce the paths used by the best ants
        
        # Find the best ant in this iteration
        current_best_idx = np.argmin(all_ant_fitnesses)
        current_best_solution = all_ant_solutions[current_best_idx]
        current_best_fitness = all_ant_fitnesses[current_best_idx]
        
        # Pheromone deposit amount (proportional to solution quality - inverse of fitness)
        # Add a small epsilon to prevent division by zero if fitness is unexpectedly low
        deposit_amount = 1.0 / (current_best_fitness + 1e-6) 
        
        # Reinforce the edges (attendee_i -> gate_j) used by the best ant
        for i in range(num_attendees):
            assigned_gate = current_best_solution[i]
            pheromones[i, assigned_gate] += deposit_amount
            
        # Optional: Normalize pheromones to prevent overflow (not strictly necessary but good practice)
        # pheromones = np.clip(pheromones, 1e-9, 100.0)

        # Early stopping for visual feedback (optional)
        if iteration % (max_iterations // 10) == 0:
            print(f"  ACO Iter {iteration}/{max_iterations}: Best Fitness = {best_fitness:.2f}")

    return best_solution
