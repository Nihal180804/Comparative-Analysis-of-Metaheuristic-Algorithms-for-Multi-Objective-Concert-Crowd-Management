import numpy as np
from crowd_management_env import CrowdManagementEnv

def solve_ga(env: CrowdManagementEnv, pop_size=50, max_generations=100, mutation_rate=0.05) -> np.ndarray:
    """
    Implements the Genetic Algorithm (GA) for gate assignment.
    """
    np.random.seed(42)
    num_attendees = env.num_attendees
    num_gates = env.num_gates
    
    # 1. Initialize Population
    population = [env.get_random_solution() for _ in range(pop_size)]
    
    for generation in range(max_generations):
        # 2. Evaluate Fitness
        fitnesses = np.array([env.calculate_solution_fitness(sol) for sol in population])
        
        best_index = np.argmin(fitnesses)
        best_solution = population[best_index]
        best_fitness = fitnesses[best_index]
        
        # 3. Selection (Elitism and Tournament)
        new_population = [best_solution.copy()]  # Elitism: keep the best solution

        # Tournament Selection
        for _ in range(pop_size - 1):
            candidates = np.random.randint(0, pop_size, 3) # Pick 3 random parents
            winner_index = candidates[np.argmin(fitnesses[candidates])]
            new_population.append(population[winner_index].copy())

        # 4. Crossover and Mutation
        offspring = []
        for i in range(0, pop_size, 2):
            parent1 = new_population[i]
            parent2 = new_population[i+1] if i + 1 < pop_size else new_population[0]
            
            # Crossover (Two-point)
            crossover_point1 = np.random.randint(1, num_attendees - 1)
            crossover_point2 = np.random.randint(crossover_point1, num_attendees)

            child1 = np.concatenate([parent1[:crossover_point1], parent2[crossover_point1:crossover_point2], parent1[crossover_point2:]])
            child2 = np.concatenate([parent2[:crossover_point1], parent1[crossover_point1:crossover_point2], parent2[crossover_point2:]])

            # Mutation
            if np.random.rand() < mutation_rate:
                mutate_index = np.random.randint(0, num_attendees)
                child1[mutate_index] = np.random.randint(0, num_gates)

            if np.random.rand() < mutation_rate:
                mutate_index = np.random.randint(0, num_attendees)
                child2[mutate_index] = np.random.randint(0, num_gates)
                
            offspring.extend([child1, child2])

        # Ensure population size remains constant (handle odd population size)
        population = offspring[:pop_size]

        # Early stopping for visual feedback (optional)
        if generation % (max_generations // 10) == 0:
            print(f"  GA Gen {generation}/{max_generations}: Best Fitness = {best_fitness:.2f}")

    return best_solution
