import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from crowd_management_env import CrowdManagementEnv, load_scenario_data
import algorithm_fcfs as fcfs
import algorithm_ga as ga
import algorithm_pso as pso
import algorithm_sa as sa
import algorithm_aco as aco


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def summarize_metrics(env: CrowdManagementEnv, solution: np.ndarray) -> dict:
    env.attendee_data['Assignment_Gate'] = solution.astype(int)
    total_queue_time, safety_violations = env._calculate_queue_metrics()
    total_distance = env._calculate_distance_metric(solution.astype(int))
    total_fitness = env.calculate_solution_fitness(solution.astype(int))

    gate_counts = pd.Series(solution.astype(int)).value_counts().sort_index()
    gate_counts = gate_counts.reindex(range(env.num_gates), fill_value=0)

    return {
        "fitness": float(total_fitness),
        "total_queue_time": float(total_queue_time),
        "total_distance": float(total_distance),
        "safety_violations": int(safety_violations),
        "gate_counts": gate_counts.to_dict()
    }


def plot_assignment(attendee_df: pd.DataFrame, gate_df: pd.DataFrame, solution: np.ndarray, title: str, outfile: str, sample_points: int = 6000):
    df = attendee_df.copy()
    df['Assigned_Gate'] = solution.astype(int)

    if len(df) > sample_points:
        df = df.sample(sample_points, random_state=42).copy()

    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(gate_df.shape[0])]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})
    ax_scatter, ax_bar = axes

    for g in range(gate_df.shape[0]):
        sub = df[df['Assigned_Gate'] == g]
        if len(sub) > 0:
            ax_scatter.scatter(sub['Arrival_Location_X'], sub['Arrival_Location_Y'], s=6, alpha=0.5, color=colors[g], label=f'Gate {g+1}')

    ax_scatter.scatter(gate_df['Location_X'], gate_df['Location_Y'], s=140, c='black', marker='X', label='Gates')
    for i, row in gate_df.reset_index(drop=True).iterrows():
        ax_scatter.text(row['Location_X'], row['Location_Y'], f"G{i+1}", fontsize=9, color='black', ha='left', va='bottom')

    ax_scatter.set_title(f"{title}\nDotted distribution (sampled)")
    ax_scatter.set_xlabel("X")
    ax_scatter.set_ylabel("Y")
    ax_scatter.legend(loc='best', fontsize=8, ncol=2, frameon=False)

    gate_counts = pd.Series(solution.astype(int)).value_counts().sort_index()
    gate_counts = gate_counts.reindex(range(gate_df.shape[0]), fill_value=0)
    ax_bar.bar([f"G{i+1}" for i in range(gate_df.shape[0])], gate_counts.values, color=[colors[i] for i in range(gate_df.shape[0])])
    ax_bar.set_title("Attendees per Gate")
    ax_bar.set_ylabel("Count")
    ax_bar.set_xticklabels([f"G{i+1}" for i in range(gate_df.shape[0])], rotation=45)

    fig.tight_layout()
    ensure_dir(os.path.dirname(outfile))
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def run_algorithms(env: CrowdManagementEnv, scenario_name: str, results_dir: str):
    num_attendees = env.num_attendees

    run_cfg = {
        "FCFS": {"run": True},
        "GA":   {"run": True, "pop_size": 40, "max_generations": 60, "mutation_rate": 0.05},
        "PSO":  {"run": True, "num_particles": 25, "max_iterations": 60, "w": 0.8, "c1": 2.0, "c2": 2.0},
        "SA":   {"run": True, "max_iterations": 1200, "initial_temp": 1000.0, "cooling_rate": 0.995},
        "ACO":  {"run": True, "num_ants": 10, "max_iterations": 50, "rho": 0.3, "alpha": 0.7, "beta": 2.0},
    }

    per_algo_results = {}
    per_algo_times = {}

    if run_cfg["FCFS"]["run"]:
        t0 = time.time()
        sol = fcfs.solve_fcfs(env)
        per_algo_times["FCFS"] = time.time() - t0
        metrics = summarize_metrics(env, sol)
        per_algo_results["FCFS"] = {"solution": sol, "metrics": metrics}

    if run_cfg["GA"]["run"]:
        t0 = time.time()
        sol = ga.solve_ga(env,
                          pop_size=run_cfg["GA"]["pop_size"],
                          max_generations=run_cfg["GA"]["max_generations"],
                          mutation_rate=run_cfg["GA"]["mutation_rate"])
        per_algo_times["GA"] = time.time() - t0
        metrics = summarize_metrics(env, sol)
        per_algo_results["GA"] = {"solution": sol, "metrics": metrics}

    if run_cfg["PSO"]["run"]:
        t0 = time.time()
        sol = pso.solve_pso(env,
                            num_particles=run_cfg["PSO"]["num_particles"],
                            max_iterations=run_cfg["PSO"]["max_iterations"],
                            w=run_cfg["PSO"]["w"],
                            c1=run_cfg["PSO"]["c1"],
                            c2=run_cfg["PSO"]["c2"])
        per_algo_times["PSO"] = time.time() - t0
        metrics = summarize_metrics(env, sol)
        per_algo_results["PSO"] = {"solution": sol, "metrics": metrics}

    if run_cfg["SA"]["run"]:
        t0 = time.time()
        sol = sa.solve_sa(env,
                          max_iterations=run_cfg["SA"]["max_iterations"],
                          initial_temp=run_cfg["SA"]["initial_temp"],
                          cooling_rate=run_cfg["SA"]["cooling_rate"])
        per_algo_times["SA"] = time.time() - t0
        metrics = summarize_metrics(env, sol)
        per_algo_results["SA"] = {"solution": sol, "metrics": metrics}

    if run_cfg["ACO"]["run"]:
        t0 = time.time()
        sol = aco.solve_aco(env,
                            num_ants=run_cfg["ACO"]["num_ants"],
                            max_iterations=run_cfg["ACO"]["max_iterations"],
                            rho=run_cfg["ACO"]["rho"],
                            alpha=run_cfg["ACO"]["alpha"],
                            beta=run_cfg["ACO"]["beta"])
        per_algo_times["ACO"] = time.time() - t0
        metrics = summarize_metrics(env, sol)
        per_algo_results["ACO"] = {"solution": sol, "metrics": metrics}

    scenario_dir = os.path.join(results_dir, scenario_name)
    ensure_dir(scenario_dir)

    summaries = []
    for algo, result in per_algo_results.items():
        sol = result["solution"]
        metrics = result["metrics"]

        plot_assignment(
            attendee_df=env.attendee_data,
            gate_df=env.gate_data,
            solution=sol,
            title=f"{scenario_name} - {algo}\nFitness={metrics['fitness']:.2f}, Queue={metrics['total_queue_time']:.2f}, Dist={metrics['total_distance']:.2f}, SafetyV={metrics['safety_violations']}",
            outfile=os.path.join(scenario_dir, f"{algo}_assignment.png"),
            sample_points=8000 if env.num_attendees > 30000 else 6000
        )

        metrics_out = metrics.copy()
        metrics_out["runtime_sec"] = per_algo_times.get(algo, None)
        with open(os.path.join(scenario_dir, f"{algo}_metrics.json"), "w") as f:
            json.dump(metrics_out, f, indent=2)

        summaries.append({
            "algorithm": algo,
            "fitness": metrics["fitness"],
            "runtime_sec": metrics_out["runtime_sec"],
            "total_queue_time": metrics["total_queue_time"],
            "total_distance": metrics["total_distance"],
            "safety_violations": metrics["safety_violations"]
        })

    if summaries:
        df_sum = pd.DataFrame(summaries).sort_values("fitness")
        best_row = df_sum.iloc[0]
        best_algo = best_row["algorithm"]
        print(f"[{scenario_name}] Best: {best_algo} | Fitness={best_row['fitness']:.2f} | Runtime={best_row['runtime_sec']:.2f}s")
        df_sum.to_csv(os.path.join(scenario_dir, "comparison_summary.csv"), index=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(df_sum["algorithm"], df_sum["fitness"], color="#3f8efc")
        ax.set_title(f"{scenario_name} - Fitness by Algorithm (lower is better)")
        ax.set_ylabel("Fitness")
        for i, v in enumerate(df_sum["fitness"]):
            ax.text(i, v, f"{v:.0f}", ha='center', va='bottom', fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(scenario_dir, "comparison_fitness.png"), dpi=140)
        plt.close(fig)

    return per_algo_results


def main():
    parser = argparse.ArgumentParser(description="Run all algorithms on existing scenarios and create metrics/plots.")
    parser.add_argument("--results-dir", default="results", help="Directory to store outputs.")
    parser.add_argument("--scenarios", nargs="*", default=None, help="Scenario names to run (default: all found).")
    args = parser.parse_args()

    base = "synthetic_datasets"
    if not os.path.isdir(base):
        print("synthetic_datasets folder not found. Please generate datasets first.")
        return

    if args.scenarios is None:
        names = []
        for fn in os.listdir(base):
            if fn.endswith("_attendee_data.csv"):
                names.append(fn.replace("_attendee_data.csv", ""))
        scenario_names = sorted(set(names))
    else:
        scenario_names = args.scenarios

    if not scenario_names:
        print("No scenarios found in synthetic_datasets.")
        return

    ensure_dir(args.results_dir)

    overall = {}
    for scenario_name in scenario_names:
        print(f"\n=== Running: {scenario_name} ===")
        attendee_df, gate_df = load_scenario_data(scenario_name)
        env = CrowdManagementEnv(attendee_df, gate_df)
        results = run_algorithms(env, scenario_name, results_dir=args.results_dir)
        summary = {}
        for algo, res in results.items():
            summary[algo] = {
                "fitness": res["metrics"]["fitness"],
                "total_queue_time": res["metrics"]["total_queue_time"],
                "total_distance": res["metrics"]["total_distance"],
                "safety_violations": res["metrics"]["safety_violations"]
            }
        overall[scenario_name] = summary

    with open(os.path.join(args.results_dir, "overall_summary.json"), "w") as f:
        json.dump(overall, f, indent=2)

    print("\nAll done. See the 'results' folder for images, per-algorithm metrics, and summaries.")


if __name__ == "__main__":
    main()