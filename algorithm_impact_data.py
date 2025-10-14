import typing

# Type definition for clarity when importing
AlgorithmImpact = typing.List[typing.Dict[str, str]]

# --- ALGORITHM_IMPACT_ANALYSIS ---
# This list contains structured data detailing the expected performance impact
# of each custom-designed synthetic scenario on the five core algorithms.

ALGORITHM_IMPACT_ANALYSIS: AlgorithmImpact = [
    {
        "Scenario": "Scenario_1_Congestion_Fail",
        "Stress_Test": "Intense, Sharp Rush",
        "FCFS": "FAILURE: Will result in maximum safety violations and longest overall wait times due to its reactive, greedy nature.",
        "SA": "POOR: May get stuck in early local minima (sub-optimal assignments) due to the intensity of the initial crowd surge.",
        "GA_PSO": "EXCELLENT: The population-based approach can schedule assignments proactively across the critical time window, minimizing the peak queue length and congestion.",
        "ACO": "GOOD: Pheromones quickly converge on the most utilized gates, helping to balance the load, but may not be as flexible as GA/PSO in multi-objective trade-offs.",
    },
    {
        "Scenario": "Scenario_2_Resource_Bottleneck",
        "Stress_Test": "Severe Gate Scarcity",
        "FCFS": "WORST: Wastes scarce capacity on nearby arrivals, leading to massive, cascading queues at all gates.",
        "SA": "GOOD: Best-suited for small, constrained spaces. Its intensive local search might find a high-quality, efficient scheduling pattern for the few available resources.",
        "GA_PSO": "EXCELLENT: Their global view is vital for scheduling the few resources efficiently. They should excel at minimizing idle time and maximizing throughput of the bottleneck gates.",
        "ACO": "AVERAGE: The limited paths/nodes means ACO's search space is small, reducing its exploration advantage. Performance may be comparable to SA.",
    },
    {
        "Scenario": "Scenario_3_Specialized_Gates",
        "Stress_Test": "Resource Heterogeneity",
        "FCFS": "AVERAGE: FCFS assigns based only on distance, completely ignoring the high-capacity 'super-gates,' leading to massive underutilization of the best resources.",
        "SA": "VULNERABLE: High risk of settling on assignments to a low-capacity gate near the arrival point (local optimum) before finding the distant, high-capacity gate (global optimum).",
        "GA_PSO": "EXCELLENT: Population methods (especially GA with its robust crossover) should easily find and exploit the high-capacity gates, leading to the best overall efficiency.",
        "ACO": "BEST: ACO's pheromones rapidly reinforce the 'path' leading to the high-capacity gates. It should show the fastest convergence toward using the most efficient resources.",
    },
    {
        "Scenario": "Scenario_4_Local_Minima_Trap",
        "Stress_Test": "High-Risk Initial Decisions",
        "FCFS": "POOR: Early sub-optimal gate assignments are compounded, resulting in consistently poor performance metrics.",
        "SA": "VITAL TEST: This scenario tests SA's high probability of accepting temporarily worse solutions to jump out of the 'trap' and find the true optimal assignment later.",
        "GA_PSO": "ROBUST: GA/PSO are less likely to fall into the trap entirely due to maintaining a large, diverse population that explores many regions simultaneously.",
        "ACO": "GOOD: Pheromones can lead to a good path, but if the initial pheromone is strong on a sub-optimal path, ACO can get stuck (stagnation).",
    },
    {
        "Scenario": "Scenario_5_Extreme_Scale",
        "Stress_Test": "Computational Scalability",
        "FCFS": "BEST RUNTIME: Will be the fastest algorithm, as it is non-iterative.",
        "SA": "VULNERABLE: Achieving convergence may take an excessively long time, making it impractical for large-scale, real-time use.",
        "GA_PSO": "RISK: The large population size required for robust optimization will severely strain memory and increase computation time. This tests the practical limits and computational cost of GA/PSO.",
        "ACO": "RISK: The graph complexity (150K nodes) increases exponentially. This tests the robustness of the evaporation/reinforcement parameters under extreme data volume.",
    },
]
