import numpy as np
import pandas as pd
from crowd_management_env import CrowdManagementEnv

def solve_fcfs(env: CrowdManagementEnv) -> np.ndarray:
    """
    Implements the First-Come-First-Served (FCFS) heuristic, which
    acts as the greedy baseline.
    
    Strategy: For each arriving group (in arrival time order), assign them to 
    the gate that currently has the *lowest estimated completion time*.
    """
    num_attendees = env.num_attendees
    num_gates = env.num_gates
    attendee_data = env.attendee_data.copy()
    gate_data = env.gate_data.copy()
    
    # Initialize gate completion time and capacity
    gate_completion_time = np.zeros(num_gates)
    gate_capacities = gate_data['Capacity_PPM'].values
    
    # Initialize the solution array
    solution = np.zeros(num_attendees, dtype=int)
    
    # The data is already sorted by Arrival_Time_Min (crucial for FCFS)
    for i in range(num_attendees):
        arrival_time = attendee_data.loc[i, 'Arrival_Time_Min']
        group_size = attendee_data.loc[i, 'Group_Size']
        
        processing_time_needed = group_size / gate_capacities
        
        # Calculate the potential completion time for each gate:
        # Time_Available = max(Current_Completion_Time, Current_Arrival_Time)
        time_available = np.maximum(gate_completion_time, arrival_time)
        
        # Completion Time = Time_Available + Processing_Time_Needed
        potential_completion_time = time_available + processing_time_needed
        
        # --- FCFS DECISION: Choose the gate that finishes the fastest ---
        best_gate_index = np.argmin(potential_completion_time)
        
        # Record the assignment and update the gate completion time
        solution[i] = best_gate_index
        gate_completion_time[best_gate_index] = potential_completion_time[best_gate_index]
        
    return solution
