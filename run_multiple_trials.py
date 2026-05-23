import os
import subprocess
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    env_name = 'lunar_lander'
    num_runs = 5
    seeds = [101, 102, 103, 104, 105]
    
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    
    print("==================================================")
    print(f"Starting {num_runs} Trials of {env_name}...")
    print("==================================================")

    # 1. Run each trial sequentially using subprocess
    for i in range(num_runs):
        run_id = str(i + 1)
        seed = seeds[i]
        log_file_path = os.path.join(log_dir, f"{env_name}_run{run_id}.log")
        
        print(f"\n[Run {run_id}/{num_runs}] Seed={seed} | Logging to {log_file_path}...")
        
        # Build command: python -u train_wann_weights.py --env smc_discrete --seed seed --run_id run_id
        cmd = [
            "python", "-u", "train_wann_weights.py",
            "--env", env_name,
            "--seed", str(seed),
            "--run_id", run_id
        ]
        
        # Run command and redirect stdout/stderr to the log file in real-time
        with open(log_file_path, "w") as log_f:
            process_result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)
            
        if process_result.returncode == 0:
            print(f"[Run {run_id}/{num_runs}] Completed successfully.")
        else:
            print(f"[Run {run_id}/{num_runs}] Failed with return code {process_result.returncode}.")
            print(f"Please inspect {log_file_path} for errors.")

    print("\n==================================================")
    print("Aggregating results and plotting shaded graphs...")
    print("==================================================")

    csv_dir = "plots/csv"
    all_runs_training = []
    all_runs_best_overall = []
    
    generations = None
    best_baseline_steps = None
    
    # 2. Parse the output CSVs from each run
    for i in range(num_runs):
        run_id = str(i + 1)
        
        # Construct CSV paths
        training_csv = os.path.join(csv_dir, f"{env_name}_training_run{run_id}.csv")
        best_overall_csv = os.path.join(csv_dir, f"{env_name}_best_overall_run{run_id}.csv")
        
        if not os.path.exists(training_csv) or not os.path.exists(best_overall_csv):
            print(f"Warning: Expected CSV files for run {run_id} are missing. Skipping aggregation for this run.")
            continue
            
        # Read training steps
        gen_list = []
        eval_steps_list = []
        with open(training_csv, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                gen_list.append(int(row[0]))
                eval_steps_list.append(float(row[1]))
                if best_baseline_steps is None:
                    best_baseline_steps = float(row[2])
                    
        # Read best overall steps
        best_steps_list = []
        with open(best_overall_csv, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                best_steps_list.append(float(row[1]))
                
        if generations is None:
            generations = np.array(gen_list)
            
        all_runs_training.append(eval_steps_list)
        all_runs_best_overall.append(best_steps_list)
        
    if len(all_runs_training) == 0:
        print("Error: No data was successfully aggregated. Make sure training completed successfully.")
        return

    # Convert to 2D numpy arrays: shape (num_runs, num_generations)
    all_runs_training = np.array(all_runs_training)
    all_runs_best_overall = np.array(all_runs_best_overall)

    # 3. Compute mean and standard deviation across runs
    mean_training = np.mean(all_runs_training, axis=0)
    std_training = np.std(all_runs_training, axis=0)
    
    mean_best_overall = np.mean(all_runs_best_overall, axis=0)
    std_best_overall = np.std(all_runs_best_overall, axis=0)

    # Make output directory for aggregated plots
    agg_plot_dir = "plots/aggregated"
    os.makedirs(agg_plot_dir, exist_ok=True)

    # 4. Plot Aggregated Policy Evaluation Curves with standard deviation shading
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(generations, mean_training, linewidth=2.5, color='#2196F3', label=f'Policy Eval (Mean)')
    ax.fill_between(generations, mean_training - std_training, mean_training + std_training, 
                    color='#2196F3', alpha=0.15, label='±1 Std Dev')
    
    if best_baseline_steps is not None:
        ax.axhline(y=best_baseline_steps, linestyle='--', linewidth=2, color='#E53935', label='Best Shared Baseline')
        
    ax.set_xlabel('Generation', fontsize=14)
    ax.set_ylabel('Average Steps', fontsize=14)
    ax.set_title(f'ES Weight Training — Steps per Generation (Aggregated {num_runs} runs)', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    png_path = os.path.join(agg_plot_dir, f"{env_name}_multi_run_eval.png")
    pdf_path = os.path.join(agg_plot_dir, f"{env_name}_multi_run_eval.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, format='pdf')
    plt.close(fig)
    print(f"Saved aggregated training plots: {png_path} and {pdf_path}")

    # 5. Plot Aggregated Best Overall ("Lowest no.steps so far") with standard deviation shading
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(generations, mean_best_overall, linewidth=2.5, color='#2196F3', label='Lowest no.steps so far (Mean)')
    ax.fill_between(generations, mean_best_overall - std_best_overall, mean_best_overall + std_best_overall, 
                    color='#2196F3', alpha=0.15, label='±1 Std Dev')
    
    if best_baseline_steps is not None:
        ax.axhline(y=best_baseline_steps, linestyle='--', linewidth=2, color='#E53935', label='Best Shared Baseline')
        
    ax.set_xlabel('Generation', fontsize=14)
    ax.set_ylabel('No. steps', fontsize=14)
    ax.set_title(f'ES Weight Training — Lowest no.steps so far (Aggregated {num_runs} runs)', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    best_png_path = os.path.join(agg_plot_dir, f"{env_name}_multi_run_best_overall.png")
    best_pdf_path = os.path.join(agg_plot_dir, f"{env_name}_multi_run_best_overall.pdf")
    fig.savefig(best_png_path, dpi=300)
    fig.savefig(best_pdf_path, format='pdf')
    plt.close(fig)
    print(f"Saved aggregated best overall plots: {best_png_path} and {best_pdf_path}")
    
    print("\nAll runs finished. Aggregated shaded plots created successfully!")

if __name__ == "__main__":
    main()
