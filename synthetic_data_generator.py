import pandas as pd
import numpy as np
import random
import os

# --- CONFIGURATION FOR MULTIPLE SCENARIOS ---
# Define five distinct concert venues (test cases) to benchmark algorithms
SCENARIOS = {
    # 1. CONGESTION FAILURE TEST (Exposes FCFS Shortcomings)
    # Goal: Create intense, catastrophic congestion where FCFS cannot recover.
    "Scenario_1_Congestion_Fail": {
        "num_attendees": 40000,
        "num_gates": 8,
        "venue_size": 1500,
        "peak_crowd_skew": 45,         # VERY LOW SKEW (Intense, sharp rush -> FCFS will fail catastrophically)
        "capacity_range": (150, 300),
    },

    # 2. BOTTLENECK SCHEDULING TEST (Highlights GA/PSO Predictive Strength)
    # Goal: Force algorithms to make critical, high-impact scheduling decisions due to resource scarcity.
    "Scenario_2_Resource_Bottleneck": {
        "num_attendees": 20000,
        "num_gates": 4,               # VERY FEW GATES (Severe resource bottleneck relative to attendance)
        "venue_size": 1000,
        "peak_crowd_skew": 90,
        "capacity_range": (200, 400),
    },

    # 3. EXPLOITATION CHALLENGE (Highlights ACO/Global Search Exploitation)
    # Goal: Test the ability to quickly find and exploit the single best resource among many.
    "Scenario_3_Specialized_Gates": {
        "num_attendees": 60000,
        "num_gates": 12,
        "venue_size": 2000,
        "peak_crowd_skew": 120,
        "capacity_range": (100, 700), # WIDE RANGE (Creates highly unequal 'specialized' resources, testing exploitation)
    },

    # 4. LOCAL MINIMA TRAP (Highlights SA's Exploration vs. GA/PSO Robustness)
    # Goal: Create a trap where simple greed leads to a local optimum, testing SA's ability to escape.
    "Scenario_4_Local_Minima_Trap": {
        "num_attendees": 15000,
        "num_gates": 6,
        "venue_size": 800,
        "peak_crowd_skew": 75,
        "capacity_range": (50, 250), # LOW CAPACITY BASELINE (makes early assignments critical)
    },

    # 5. EXTREME SCALABILITY TEST (Tests Computational Time and Robustness)
    # Goal: Benchmark the runtime and memory usage of population-based methods (GA, PSO).
    "Scenario_5_Extreme_Scale": {
        "num_attendees": 150000,      # VERY LARGE NUMBER
        "num_gates": 25,
        "venue_size": 4000,
        "peak_crowd_skew": 180,       # Very long, distributed arrival window
        "capacity_range": (400, 700),
    }
}

# --- GLOBAL CONSTANTS ---
CONCERT_START_TIME_MIN = 300 # Concert starts at T=300 minutes (5 hours from opening)

def generate_venue_data(scenario_name, config):
    """
    Generates static data for the entry gates based on scenario config.
    """
    num_gates = config["num_gates"]
    venue_size = config["venue_size"]
    cap_min, cap_max = config["capacity_range"]
    
    print(f"[{scenario_name}] Generating data for {num_gates} gates...")
    gates_data = {
        'Gate_ID': [f'G{i+1}' for i in range(num_gates)],
        'Location_X': np.random.uniform(0, venue_size, num_gates),
        'Location_Y': np.random.uniform(0, venue_size, num_gates),
        'Capacity_PPM': np.random.randint(cap_min, cap_max, num_gates) # Capacity Pts/Min
    }
    return pd.DataFrame(gates_data)

def generate_attendee_data(scenario_name, config):
    """
    Generates dynamic data for individual attendees based on scenario config.
    """
    num_attendees = config["num_attendees"]
    venue_size = config["venue_size"]
    
    # Calculate peak arrival time based on a standard offset from concert start
    peak_time_min = CONCERT_START_TIME_MIN - 30 
    
    print(f"[{scenario_name}] Generating data for {num_attendees} attendees...")
    attendee_ids = [f'{scenario_name[0]}{i+1}' for i in range(num_attendees)] # Unique ID based on scenario

    # 1. Group Size Distribution (Weighted to have more small groups)
    group_sizes = np.random.choice([1, 2, 3, 4, 5], 
                                   size=num_attendees, 
                                   p=[0.50, 0.30, 0.10, 0.05, 0.05])

    # 2. Arrival Time Distribution (Normal distribution centered near peak time)
    # The 'peak_crowd_skew' determines the standard deviation (tightness of the rush)
    arrival_times = np.random.normal(loc=peak_time_min, 
                                     scale=config["peak_crowd_skew"], 
                                     size=num_attendees).astype(int)
    
    # Ensure no negative arrival times
    arrival_times = np.maximum(0, arrival_times)
    
    # 3. Arrival Location (Random scatter around the venue area)
    arrival_x = np.random.uniform(0, venue_size, num_attendees)
    arrival_y = np.random.uniform(0, venue_size, num_attendees)

    attendee_data = {
        'Attendee_ID': attendee_ids,
        'Group_Size': group_sizes,
        'Arrival_Time_Min': arrival_times,
        'Arrival_Location_X': arrival_x,
        'Arrival_Location_Y': arrival_y,
    }

    # Sort by arrival time to simulate the sequence of arrival events
    return pd.DataFrame(attendee_data).sort_values(by='Arrival_Time_Min').reset_index(drop=True)

def calculate_distances(attendee_df, gate_df, scenario_name):
    """
    Calculates the Euclidean distance from every attendee's arrival to every gate.
    """
    print(f"[{scenario_name}] Calculating distance matrix...")
    
    num_gates = len(gate_df)
    distance_cols = [f'Dist_G{i+1}' for i in range(num_gates)]
    distances = pd.DataFrame(index=attendee_df.index, columns=distance_cols)

    # Calculate Euclidean distance for each attendee-gate pair
    for i, gate in gate_df.iterrows():
        gate_col = distance_cols[i]
        # Use vectorized operations for speed
        distances[gate_col] = np.sqrt(
            (attendee_df['Arrival_Location_X'] - gate['Location_X'])**2 +
            (attendee_df['Arrival_Location_Y'] - gate['Location_Y'])**2
        )

    return pd.concat([attendee_df, distances], axis=1)

def run_scenario(scenario_name, config):
    """Runs the generation process for a single scenario."""
    print(f"\n--- STARTING SCENARIO: {scenario_name} ---")
    
    # Set a unique seed for this scenario to ensure reproducibility
    seed = hash(scenario_name) % 2**32
    np.random.seed(seed)
    random.seed(seed)
    
    # 1. Generate Gate Data
    gate_df = generate_venue_data(scenario_name, config)

    # 2. Generate Attendee Data
    attendee_df = generate_attendee_data(scenario_name, config)

    # 3. Calculate Distance Matrix
    final_df = calculate_distances(attendee_df, gate_df, scenario_name)

    # 4. Save Output Files
    output_dir = "synthetic_datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    gate_file = os.path.join(output_dir, f'{scenario_name}_gate_data.csv')
    attendee_file = os.path.join(output_dir, f'{scenario_name}_attendee_data.csv')
    
    gate_df.to_csv(gate_file, index=False)
    final_df.to_csv(attendee_file, index=False)

    print(f"[{scenario_name}] Data saved to directory: {output_dir}")
    print(f"[{scenario_name}] {len(final_df)} attendees and {len(gate_df)} gates generated.")
    
    return final_df, gate_df

def main():
    """Main function to run all defined scenarios."""
    all_results = {}
    for name, config in SCENARIOS.items():
        attendee_data, gate_data = run_scenario(name, config)
        all_results[name] = (attendee_data, gate_data)

    print("\n--- ALL SCENARIOS COMPLETE ---")
    print(f"A total of {len(SCENARIOS)} scenarios were generated and saved in the 'synthetic_datasets' folder.")
    print("These datasets are designed to rigorously test all your proposed algorithms.")

if __name__ == "__main__":
    main()
