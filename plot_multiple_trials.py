import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_results(env_name, num_runs, csv_dir="plots/csv", agg_plot_dir="plots/aggregated"):
    """
    Reads CSV logs for a given environment across multiple runs, computes mean and standard
    deviation, and generates aggregated shaded plots (PNG and PDF).
    """
    print("\n==================================================")
    print(f"Aggregating results and plotting shaded graphs for {env_name}...")
    print("==================================================")

    all_runs_training = []
    all_runs_best_overall = []
    
    generations = None
    best_baseline_steps = None
    
    # 1. Parse the output CSVs from each run
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
        return False

    # Convert to 2D numpy arrays: shape (num_runs, num_generations)
    all_runs_training = np.array(all_runs_training)
    all_runs_best_overall = np.array(all_runs_best_overall)

    # 2. Compute mean and standard deviation across runs
    mean_training = np.mean(all_runs_training, axis=0)
    std_training = np.std(all_runs_training, axis=0)
    
    mean_best_overall = np.mean(all_runs_best_overall, axis=0)
    std_best_overall = np.std(all_runs_best_overall, axis=0)

    # Make output directory for aggregated plots
    os.makedirs(agg_plot_dir, exist_ok=True)

    # 3. Plot Aggregated Policy Evaluation Curves with standard deviation shading
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(generations, mean_training, linewidth=5.0, color='#2196F3', label='Policy Eval (Mean)')
    ax.fill_between(generations, mean_training - std_training, mean_training + std_training, 
                    color='#2196F3', alpha=0.15, label='±1 Std Dev')
    
    if best_baseline_steps is not None:
        ax.axhline(y=best_baseline_steps, linestyle='--', linewidth=4.0, color='#E53935', label='Shared-Weight Baseline')
        
    ax.set_xlabel('Generation', fontsize=22)
    ax.set_ylabel('Average Steps', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(loc='upper right', fontsize=18)
    ax.grid(True, alpha=0.3)
    if env_name == 'smc_continuous':
        ax.set_ylim(100, 140)
    elif env_name == 'smc_discrete':
        ax.set_ylim(115, 155)
    elif env_name == 'lunar_lander':
        ax.set_ylim(55, 95)
    fig.tight_layout()
    
    png_path = os.path.join(agg_plot_dir, f"{env_name}_multi_run_eval.png")
    pdf_path = os.path.join(agg_plot_dir, f"{env_name}_multi_run_eval.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, format='pdf')
    plt.close(fig)
    print(f"Saved aggregated training plots: {png_path} and {pdf_path}")

    # 4. Plot Aggregated Best Overall ("Lowest no.steps so far") with standard deviation shading
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(generations, mean_best_overall, linewidth=5.0, color='#2196F3', label='Evolved weights')
    ax.fill_between(generations, mean_best_overall - std_best_overall, mean_best_overall + std_best_overall, 
                    color='#2196F3', alpha=0.15, label='±1 Std Dev')
    
    if best_baseline_steps is not None:
        ax.axhline(y=best_baseline_steps, linestyle='--', linewidth=4.0, color='#E53935', label='Shared-Weight Baseline')
        
    ax.set_xlabel('Generation', fontsize=22)
    ax.set_ylabel('No. steps', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(loc='upper right', fontsize=18)
    ax.grid(True, alpha=0.3)
    if env_name == 'smc_continuous':
        ax.set_ylim(100, 140)
    elif env_name == 'smc_discrete':
        ax.set_ylim(115, 155)
    elif env_name == 'lunar_lander':
        ax.set_ylim(55, 95)
    fig.tight_layout()
    
    best_png_path = os.path.join(agg_plot_dir, f"{env_name}_multi_run_best_overall.png")
    best_pdf_path = os.path.join(agg_plot_dir, f"{env_name}_multi_run_best_overall.pdf")
    fig.savefig(best_png_path, dpi=300)
    fig.savefig(best_pdf_path, format='pdf')
    plt.close(fig)
    print(f"Saved aggregated best overall plots: {best_png_path} and {best_pdf_path}")
    
    print(f"\nAggregated shaded plots for {env_name} created successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Plot Aggregated Curves from Multiple WANN Runs")
    parser.add_argument('--env', type=str, default='smc_discrete', 
                        choices=['smc_discrete', 'smc_continuous', 'lunar_lander'],
                        help="Environment name (default: smc_discrete)")
    parser.add_argument('--num_runs', type=int, default=5,
                        help="Number of runs/trials to aggregate (default: 5)")
    parser.add_argument('--csv_dir', type=str, default='plots/csv',
                        help="Directory containing the run CSV files (default: plots/csv)")
    parser.add_argument('--agg_plot_dir', type=str, default='plots/aggregated',
                        help="Directory where output plots will be saved (default: plots/aggregated)")
    
    args = parser.parse_args()
    
    success = plot_results(
        env_name=args.env,
        num_runs=args.num_runs,
        csv_dir=args.csv_dir,
        agg_plot_dir=args.agg_plot_dir
    )
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
