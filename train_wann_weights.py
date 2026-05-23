
import os
import time
import csv
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from convert_wann import WannPyTorch
from domain.sparse_mountain_car import SparseMountainCarEnv
from domain.sparse_mountain_car_conti import SparseMountainCarContiEnv
from domain.lunar_lander import LunarLanderEnv

# Environment configurations: weight_file, env_class, n_input, n_output, continuous
ENV_CONFIGS = {
    'smc_discrete':   ('champions/smc_best.out',       SparseMountainCarEnv,      2, 3, False),
    'smc_continuous':  ('champions/smc_conti_best.out', SparseMountainCarContiEnv, 2, 1, True),
    'lunar_lander':    ('champions/lula_best.out',      LunarLanderEnv,            8, 4, False),
}

# Helper for running episodes
def _run_episodes(n_episodes, env_cls=None, continuous=False, n_input=None, n_output=None,
                  weight_file=None, edge_weights_np=None, shared_weight=None, trainable=True, seed=None):
    """
    Run n_episodes and return (avg_steps, avg_reward, all_steps).
    """
    try:
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Initialize model
        model = WannPyTorch(weight_file, n_input, n_output, trainable=trainable)
        # Overwrite model weights with edge_weights_np if trainable
        if trainable:
            with torch.no_grad():
                model.edge_weights.copy_(torch.from_numpy(edge_weights_np))

        env = env_cls(render_mode=None)

        # Run the simulation for n_episodes
        total_steps = 0
        total_reward = 0
        all_steps = []
        for _ in range(n_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            done = False
            steps = 0
            state_tensor = torch.from_numpy(state).float()
            while not done:
                with torch.no_grad():
                    # If network is trainable use all different weights from edge_weights_np
                    # Otherwise use one single shared weight
                    if trainable:
                        output = model(state_tensor)
                    else:
                        output = model(state_tensor, weight=shared_weight)
                
                if continuous:
                    action = output.numpy()
                else:
                    action = torch.argmax(output).item()
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1
                state_tensor = torch.from_numpy(next_state).float()
                total_reward += reward
            total_steps += steps
            all_steps.append(steps)
        
        return total_steps / n_episodes, total_reward / n_episodes, all_steps
    except Exception as e:
        print(f"Worker Error: {e}")
        import traceback
        traceback.print_exc()
        return 200.0, 0.0, [200.0]*n_episodes

# Training Script

def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="ES WANN Weight Training Script")
    parser.add_argument('--env', type=str, default='lunar_lander', choices=['smc_discrete', 'smc_continuous', 'lunar_lander'],
                        help="Environment configurations")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument('--run_id', type=str, default="", help="Run identifier/suffix for output files")
    args = parser.parse_args()

    ENV_NAME = args.env
    
    # Apply random seed to the parent process if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Set parent process random seed: {args.seed}")

    suffix = f"_run{args.run_id}" if args.run_id else ""

    WEIGHT_FILE, ENV_CLS, N_INPUT, N_OUTPUT, CONTINUOUS = ENV_CONFIGS[ENV_NAME]
    print(f"Environment: {ENV_NAME}  (inputs={N_INPUT}, outputs={N_OUTPUT}, continuous={CONTINUOUS}){f' | Run: {args.run_id}' if args.run_id else ''}")
    
    POP_SIZE = 16         # 16 pairs = 32 individuals
    GENERATIONS = 100
    LEARNING_RATE = 0.05
    SIGMA = 0.05          
    N_EVALS_PER_CAND = 100
    
    print("--- 0. Shared Weight Baseline ---")
    w_vals = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
    best_shared_steps = 200.0
    best_shared_weight = None
    best_shared_steps_list = None
    
    for w in w_vals:
        avg_steps, avg_reward, steps_list = _run_episodes(n_episodes=200, env_cls=ENV_CLS, continuous=CONTINUOUS, n_input=N_INPUT, n_output=N_OUTPUT, weight_file=WEIGHT_FILE, shared_weight=w, trainable=False)
        std_dev = np.std(steps_list)
        print(f"Weight {w: .1f}: Avg Steps={avg_steps:.1f} ± {std_dev:.1f}, Avg Reward={avg_reward:.1f}")
        if avg_steps < best_shared_steps:
            best_shared_steps = avg_steps
            best_shared_weight = w
            best_shared_steps_list = steps_list            
    print(f"Best Shared Weight Baseline: {best_shared_weight:.1f} with {best_shared_steps:.1f} steps.\n")

    print("--- 1. ES Training ---")

    # Create a master tmp model to track weights update
    tmp_model = WannPyTorch(WEIGHT_FILE, n_input=N_INPUT, n_output=N_OUTPUT, trainable=True)
    nn.init.constant_(tmp_model.edge_weights, best_shared_weight)
    current_weights = tmp_model.edge_weights.detach().cpu().numpy()
    # best_overall tracks the best master policy weights (evolved via ES) evaluated across all generations
    best_overall_steps = 200.0
    best_overall_weights = current_weights.copy()

    # best_mutant tracks the best individual mutant candidate evaluated across all generations
    best_mutant_steps = 200.0
    best_mutant_weights = current_weights.copy()

    # History tracking for plots
    history_avg_steps = []
    history_avg_reward = []
    history_eval_steps = []
    history_eval_reward = []
    history_best_candidate_steps = []
    history_best_overall_steps = []

    
    print(f"Starting Advanced ES Training on {os.cpu_count()} cores...")

    # Run parallel tasks
    with ProcessPoolExecutor() as executor:
        training_start_time = time.time()
        for gen in range(GENERATIONS):
            start_time = time.time()

            # Generate noise vectors for each individual of the population (POP_SIZE)
            adj_mask = tmp_model.adjacency.cpu().numpy()

            noise_vectors = [np.random.randn(*current_weights.shape).astype(np.float32) * adj_mask for _ in range(POP_SIZE)]
            
            # For every noise vector, generate two candidates (positive and negative mutants)
            candidates = []
            for epsilon in noise_vectors:
                candidates.append(current_weights + SIGMA * epsilon) # Positive mutant
                candidates.append(current_weights - SIGMA * epsilon) # Negative mutant
        
            futures = []
            for cand in candidates:
                futures.append(executor.submit(_run_episodes, N_EVALS_PER_CAND, env_cls=ENV_CLS, continuous=CONTINUOUS,
                                n_input=N_INPUT, n_output=N_OUTPUT, weight_file=WEIGHT_FILE, edge_weights_np=cand, trainable=True))
            
            # OpenAI ES Update Logic (Vectorized)
            # For each candidate extract the average step count
            results_np_s = np.array([f.result()[0] for f in futures])
            results_np_r = np.array([f.result()[1] for f in futures])
            
            # Order each candidate based on the steps taken, the best (lowest steps) -> Rank 0
            ranks = np.empty(len(results_np_r), dtype=int)
            ranks[np.argsort(results_np_r)] = np.arange(len(results_np_r)) # ranks is an array where to each candidate [0, POP_SIZE*2] is assigned a rank

            # Utilities: map ranks to [-0.5, 0.5], where better performers get higher utility
            # (Note: lower steps is better, so Rank 0 should get 0.5)
            utilities = np.linspace(-0.5, 0.5, num=len(results_np_r)) # array of size POP_SIZE*2 going from 0.5 to -0.5
            fitness = utilities[ranks] # array where each candidate has a fitness value decided by its rank

            # Separate utilities for positive and negative mutants
            # candidates = [pos1, neg1, pos2, neg2, ...]
            u_pos = fitness[0::2]
            u_neg = fitness[1::2]
            
            # Weighted sum of noise vectors
            # update = sum( (u_pos - u_neg) * epsilon )
            # u_pos - u_neg is of size POP_SIZE
            # noise_vectors is of size POP_SIZE * W
            # update is of size W
            # This is a weighted sum of the POP_SIZE noise matrices, with u_pos - u_neg as weight
            update = np.tensordot(u_pos - u_neg, noise_vectors, axes=1)

            current_weights += (LEARNING_RATE / (POP_SIZE * SIGMA)) * update
            
            gen_min = np.min(results_np_s)
            gen_avg = np.mean(results_np_s)
            gen_avg_reward = np.mean(results_np_r)
            
            # Mutant-based best tracking
            if gen_min < best_mutant_steps:
                best_mutant_steps = gen_min
                best_idx = np.argmin(results_np_s)
                best_mutant_weights = candidates[best_idx].copy()
            
            # Evaluate current policy with updated weights (100 episodes)
            eval_future = executor.submit(_run_episodes, 100, env_cls=ENV_CLS, continuous=CONTINUOUS,
                            n_input=N_INPUT, n_output=N_OUTPUT, weight_file=WEIGHT_FILE,
                            edge_weights_np=current_weights.copy(), trainable=True)
            eval_steps, eval_reward, _ = eval_future.result()

            # Policy-based best tracking
            if eval_steps < best_overall_steps:
                best_overall_steps = eval_steps
                best_overall_weights = current_weights.copy()
            
            # Track history
            history_avg_steps.append(gen_avg)
            history_avg_reward.append(gen_avg_reward)
            history_eval_steps.append(eval_steps)
            history_eval_reward.append(eval_reward)
            history_best_candidate_steps.append(gen_min)
            history_best_overall_steps.append(best_overall_steps)
            
            duration = time.time() - start_time
            
            # Progress bar & ETA calculation
            elapsed_training = time.time() - training_start_time
            avg_time_per_gen = elapsed_training / (gen + 1)
            gens_remaining = GENERATIONS - (gen + 1)
            eta_seconds = avg_time_per_gen * gens_remaining
            
            if eta_seconds < 60:
                eta_str = f"{eta_seconds:.0f}s"
            elif eta_seconds < 3600:
                eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            else:
                eta_str = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m"
                
            print(f"Gen {gen + 1:03d}/{GENERATIONS} | Pop Avg: {gen_avg:.1f} | Updated Policy: {eval_steps:.1f} | Best Policy: {best_overall_steps:.1f} | Best Mutant: {best_mutant_steps:.1f} | ETA: {eta_str} (Time: {duration:.2f}s)")
            
            if best_overall_steps < 40: 
                print("Task solved! Stopping training.")
                break

    print("\n--- 2. Final Trained Network Verification ---")
    
    # 1. Evaluate the best policy weights (best_overall_weights) over 200 episodes
    print("Evaluating final policy weights...")
    final_policy_steps, final_policy_reward, final_policy_list = _run_episodes(
        200, env_cls=ENV_CLS, continuous=CONTINUOUS, weight_file=WEIGHT_FILE,
        n_input=N_INPUT, n_output=N_OUTPUT, edge_weights_np=best_overall_weights
    )
    
    # 2. Evaluate the best mutant weights (best_mutant_weights) over 200 episodes
    print("Evaluating best mutant weights...")
    final_mutant_steps, final_mutant_reward, final_mutant_list = _run_episodes(
        200, env_cls=ENV_CLS, continuous=CONTINUOUS, weight_file=WEIGHT_FILE,
        n_input=N_INPUT, n_output=N_OUTPUT, edge_weights_np=best_mutant_weights
    )
    
    std_base = np.std(best_shared_steps_list)
    std_policy = np.std(final_policy_list)
    std_mutant = np.std(final_mutant_list)
    
    # Welch's t-test comparisons
    t_stat_p, p_value_p = stats.ttest_ind(best_shared_steps_list, final_policy_list, equal_var=False)
    t_stat_m, p_value_m = stats.ttest_ind(best_shared_steps_list, final_mutant_list, equal_var=False)

    print("\n==================== FINAL COMPARISON ====================")
    print(f"Best Shared Baseline (200 episodes):        Avg Steps = {best_shared_steps:.1f} ± {std_base:.1f}")
    print(f"Best Mutant (200 episodes):     Avg Steps = {final_mutant_steps:.1f} ± {std_mutant:.1f}, Avg Reward = {final_mutant_reward:.1f}")
    print(f"Best Policy (200 episodes):     Avg Steps = {final_policy_steps:.1f} ± {std_policy:.1f}, Avg Reward = {final_policy_reward:.1f}")
    print("----------------------------------------------------------")
    print(f"Welch's t-test p-value (Baseline vs Mutant): {p_value_m:.6f}")
    print(f"Welch's t-test p-value (Baseline vs Policy): {p_value_p:.6f}")
    print("==========================================================\n")
    
    best_final_steps = min(final_policy_steps, final_mutant_steps)
    if best_final_steps < best_shared_steps:
        print(f"Success! Performance improved from {best_shared_steps:.1f} to {best_final_steps:.1f} steps.")
    else:
        print("Final verification did not show improvement over baseline.")

    # Plot Training Curves
    plot_dir = "plots"
    latex_dir = os.path.join(plot_dir, "Latex")
    png_dir = os.path.join(plot_dir, "png")
    os.makedirs(latex_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    generations = list(range(len(history_avg_steps)))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(generations, history_eval_steps, linewidth=2, color='#2196F3', label='Policy Eval Steps (100 runs)')
    ax.axhline(y=best_shared_steps, linestyle='--', linewidth=2, color='#E53935', label='Shared-Weight Baseline')
    ax.set_xlabel('Generation', fontsize=14)
    ax.set_ylabel('Average Steps', fontsize=14)
    ax.set_title('ES Weight Training — Steps per Generation', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    if ENV_NAME == 'smc_continuous':
        ax.set_ylim(100, 140)
    elif ENV_NAME == 'smc_discrete':
        ax.set_ylim(115, 155)
    elif ENV_NAME == 'lunar_lander':
        ax.set_ylim(55, 95)
    fig.tight_layout()
    fig.savefig(os.path.join(latex_dir, f'{ENV_NAME}_training{suffix}.pdf'), format='pdf')
    fig.savefig(os.path.join(png_dir, f'{ENV_NAME}_training{suffix}.png'), format='png')
    plt.close(fig)
    print(f"Saved: {latex_dir}/{ENV_NAME}_training{suffix}.pdf and {png_dir}/{ENV_NAME}_training{suffix}.png")

    # Plot Best Overall steps ("Lowest no.steps so far")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(generations, history_best_overall_steps, linewidth=2, color='#2196F3', label='Lowest no.steps so far')
    ax.axhline(y=best_shared_steps, linestyle='--', linewidth=2, color='#E53935', label='Shared-Weight Baseline')
    ax.set_xlabel('Generation', fontsize=14)
    ax.set_ylabel('No. steps', fontsize=14)
    ax.set_title('ES Weight Training — Lowest no.steps so far', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    if ENV_NAME == 'smc_continuous':
        ax.set_ylim(100, 140)
    elif ENV_NAME == 'smc_discrete':
        ax.set_ylim(115, 155)
    elif ENV_NAME == 'lunar_lander':
        ax.set_ylim(55, 95)
    fig.tight_layout()
    fig.savefig(os.path.join(latex_dir, f'{ENV_NAME}_best_overall{suffix}.pdf'), format='pdf')
    fig.savefig(os.path.join(png_dir, f'{ENV_NAME}_best_overall{suffix}.png'), format='png')
    plt.close(fig)
    print(f"Saved: {latex_dir}/{ENV_NAME}_best_overall{suffix}.pdf and {png_dir}/{ENV_NAME}_best_overall{suffix}.png")

    # Save to CSV
    csv_dir = os.path.join(plot_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    
    # Save training steps CSV
    training_csv_path = os.path.join(csv_dir, f'{ENV_NAME}_training{suffix}.csv')
    with open(training_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Generation', 'Policy_Eval_Steps', 'Best_Baseline_Steps'])
        for gen, val in zip(generations, history_eval_steps):
            writer.writerow([gen, val, best_shared_steps])
    print(f"Saved: {training_csv_path}")

    # Save best candidate steps CSV
    best_candidate_csv_path = os.path.join(csv_dir, f'{ENV_NAME}_best_candidate{suffix}.csv')
    with open(best_candidate_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Generation', 'Best_Candidate_Steps', 'Best_Baseline_Steps'])
        for gen, val in zip(generations, history_best_candidate_steps):
            writer.writerow([gen, val, best_shared_steps])
    print(f"Saved: {best_candidate_csv_path}")
            
    # Save best overall steps CSV
    best_overall_csv_path = os.path.join(csv_dir, f'{ENV_NAME}_best_overall{suffix}.csv')
    with open(best_overall_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Generation', 'Best_Overall_Steps', 'Best_Baseline_Steps'])
        for gen, val in zip(generations, history_best_overall_steps):
            writer.writerow([gen, val, best_shared_steps])
    print(f"Saved: {best_overall_csv_path}")

    
if __name__ == "__main__":
    main()
