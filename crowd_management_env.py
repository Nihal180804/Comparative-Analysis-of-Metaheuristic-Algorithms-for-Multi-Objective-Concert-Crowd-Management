import numpy as np
import pandas as pd
import time
import os
from typing import List, Tuple, Dict

# --- CONFIGURATION CONSTANTS ---
SAFETY_THRESHOLD_PPM = 500  # Max people capacity per minute * capacity_multiplier
SAFETY_VIOLATION_PENALTY = 1000000.0 # Large penalty for violating safety constraint
QUEUE_TIME_WEIGHT = 0.6     # Weight for minimizing total queue time
DISTANCE_WEIGHT = 0.4       # Weight for minimizing total walking distance

class CrowdManagementEnv:
    """
    Core environment used by all algorithms to evaluate solutions.
    Handles calculating objective values (Fitness) and enforcing constraints.
    """
    def __init__(self, attendee_data: pd.DataFrame, gate_data: pd.DataFrame):
        self.attendee_data = attendee_data.copy()
        self.gate_data = gate_data.copy()
        self.num_attendees = len(attendee_data)
        self.num_gates = len(gate_data)

        # Pre-calculate distances from each attendee to each gate
        self.distance_matrix = self._get_distance_matrix()

        # Maximum cumulative capacity over the main arrival window, used for safety checks
        self.max_cumulative_capacity = self.gate_data['Capacity_PPM'].sum() * 1.5

        # Initialize the 'Assignment_Gate' column, which stores the current solution
        self.attendee_data['Assignment_Gate'] = 0 
        
    def _get_distance_matrix(self) -> np.ndarray:
        """Extracts the distance matrix from the attendee data."""
        dist_cols = [f'Dist_G{i+1}' for i in range(self.num_gates)]
        return self.attendee_data[dist_cols].values

    def calculate_solution_fitness(self, solution: np.ndarray) -> float:
        """
        Calculates the single-objective fitness value for a given solution (array of gate indices).
        
        The objective is: Minimize (Weighted Queue Time + Weighted Distance + Safety Penalty)
        """
        # Assign the solution (gate index) to the data frame
        self.attendee_data['Assignment_Gate'] = solution.astype(int)

        # 1. Calculate Queue Times (Efficiency & Safety)
        queue_time_total, safety_violation_count = self._calculate_queue_metrics()

        # 2. Calculate Distance (Experience)
        distance_total = self._calculate_distance_metric(solution.astype(int))

        # 3. Apply Multi-Objective Weights
        weighted_efficiency = queue_time_total * QUEUE_TIME_WEIGHT
        weighted_experience = distance_total * DISTANCE_WEIGHT

        # 4. Apply Safety Penalty (Hard Constraint)
        safety_penalty = safety_violation_count * SAFETY_VIOLATION_PENALTY

        # Total Fitness (to be minimized)
        total_fitness = weighted_efficiency + weighted_experience + safety_penalty

        return total_fitness

    def _calculate_queue_metrics(self) -> Tuple[float, int]:
        """
        Calculates total queue time and the number of safety violations.
        This simulates the flow and congestion at each minute.
        """
        # Group attendees by assigned gate and minute of arrival
        grouped = self.attendee_data.groupby(['Assignment_Gate', 'Arrival_Time_Min'])
        
        # Initialize gate state: (time_available, queue_size)
        gate_state = {}
        for i in range(self.num_gates):
            gate_state[i] = {'time_available': 0.0, 'queue_size': 0, 'capacity_ppm': self.gate_data.loc[i, 'Capacity_PPM']}

        total_queue_time = 0.0
        safety_violation_count = 0
        
        # Iterate through time steps (minutes) from earliest arrival
        min_time = self.attendee_data['Arrival_Time_Min'].min()
        max_time = self.attendee_data['Arrival_Time_Min'].max()
        
        # Check safety (over-capacity) at each minute
        for t in range(min_time, max_time + 10): # Check a few minutes past the last arrival
            
            # 1. Process existing queue/capacity limits for each gate
            for gate_idx, state in gate_state.items():
                capacity = state['capacity_ppm']
                
                # Check for safety constraint based on instantaneous arrival/queue pressure
                if state['queue_size'] > capacity * 1.5:
                     safety_violation_count += 1
                
                # The gate processes up to its capacity if there's a queue
                processed = min(state['queue_size'], capacity)
                state['queue_size'] -= processed

            # 2. Add new arrivals to the queue
            if t in self.attendee_data['Arrival_Time_Min'].values:
                # Get all attendees arriving at this minute
                new_arrivals = self.attendee_data[self.attendee_data['Arrival_Time_Min'] == t]
                
                for gate_idx in range(self.num_gates):
                    arrivals_at_gate = new_arrivals[new_arrivals['Assignment_Gate'] == gate_idx]
                    
                    if not arrivals_at_gate.empty:
                        arrival_groups = arrivals_at_gate['Group_Size'].sum()
                        
                        # Simple model: arrivals add to the queue size
                        gate_state[gate_idx]['queue_size'] += arrival_groups
                        
                        # Add queue time based on current time_available
                        current_time_available = gate_state[gate_idx]['time_available']
                        
                        for index, attendee in arrivals_at_gate.iterrows():
                            # Entry time is the current gate's availability time
                            entry_time = max(current_time_available, attendee['Arrival_Time_Min'])
                            total_queue_time += (entry_time - attendee['Arrival_Time_Min'])
                        
                        # Update gate availability time based on new arrivals
                        processing_time_needed = arrival_groups / gate_state[gate_idx]['capacity_ppm']
                        
                        gate_state[gate_idx]['time_available'] = max(t, current_time_available) + processing_time_needed


        return total_queue_time, safety_violation_count

    def _calculate_distance_metric(self, solution: np.ndarray) -> float:
        """Calculates the total distance traveled by all attendees."""
        
        total_distance = 0.0
        
        # Iterate through attendees
        for i in range(self.num_attendees):
            assigned_gate_index = solution[i]
            # Use the pre-calculated distance matrix
            total_distance += self.distance_matrix[i, assigned_gate_index]
            
        return total_distance
    
    def get_random_solution(self) -> np.ndarray:
        """Returns a random gate assignment solution."""
        return np.random.randint(0, self.num_gates, self.num_attendees)
    
    def get_distance_matrix(self) -> np.ndarray:
        """Exposes the pre-calculated distance matrix."""
        return self.distance_matrix


# Example helper function to load data (used by testing framework)
def load_scenario_data(scenario_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads attendee and gate data from the synthetic_datasets folder."""
    base_path = "synthetic_datasets"
    
    # Check if the necessary files exist (Assuming user has run the generator)
    attendee_file = os.path.join(base_path, f'{scenario_name}_attendee_data.csv')
    gate_file = os.path.join(base_path, f'{scenario_name}_gate_data.csv')

    if not os.path.exists(attendee_file) or not os.path.exists(gate_file):
        raise FileNotFoundError(
            f"Required files not found. Please ensure you have run the data generator script to create the 'synthetic_datasets' folder and files like '{scenario_name}_attendee_data.csv'."
        )

    attendee_data = pd.read_csv(attendee_file)
    gate_data = pd.read_csv(gate_file)
    
    # Ensure gate_data is indexed by gate ID (0 to num_gates-1)
    gate_data.reset_index(drop=True, inplace=True) 

    return attendee_data, gate_data
