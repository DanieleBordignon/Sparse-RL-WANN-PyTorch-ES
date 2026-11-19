
import os
import time
import csv
import random
import argparse
import itertools
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

# Environment configurations: weight_file, env_class, n_input, n_output, continuous, max_steps
# max_steps = episode length limit of the environment (max_episode_length in domain/config.py).
# Episodes that do not reach the goal are counted as max_steps (see _run_episodes).
ENV_CONFIGS = {
    'smc_discrete':   ('champions/smc_best.out',       SparseMountainCarEnv,      2, 3, False, 200),
    'smc_continuous':  ('champions/smc_conti_best.out', SparseMountainCarContiEnv, 2, 1, True,  200),
    # NOTE: champions/lula_best.out in the original repo is the weaker run (lula_1024_128) and never lands.
    # The network the original authors actually report is lula_256_480 (2 hidden nodes, 17 connections).
    'lunar_lander':    ('log/lula/lula_256_480_best.out', LunarLanderEnv,          8, 4, False, 1000),
}

# Helper for running episodes
def _run_episodes(n_episodes, env_cls=None, continuous=False, n_input=None, n_output=None,
                  weight_file=None, edge_weights_np=None, shared_weight=None, trainable=True, seed=None,
                  max_steps=None):
    """
    Run n_episodes and return (avg_steps, avg_reward, all_steps, outcomes, episodes).
    episodes lists, for each episode, (raw_steps, counted_steps, episode_reward, outcome).
    If max_steps is given, every episode that does not end in success (fail or timeout) is
    counted as max_steps in avg_steps and all_steps, so failing is always the worst result.
    (For Sparse Mountain Car this changes nothing: a failed episode already lasts max_steps=200.)
    If seed is given, the episodes are reproducible (see the env.reset call below).
    outcomes counts how each episode ended:
      'success': terminated with a positive reward (goal reached / safe landing)
      'fail':    terminated with zero reward (e.g. crash or out of bounds)
      'timeout': truncated at the environment's step limit
      'error':   episodes not run because of an exception
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
        outcomes = {'success': 0, 'fail': 0, 'timeout': 0, 'error': 0}
        episodes = []  # per-episode details: (raw_steps, counted_steps, episode_reward, outcome)
        for ep in range(n_episodes):
            # Seed the environment only on the first episode: the following episodes continue the
            # environment's own random sequence, so they differ from each other but are reproducible.
            # (Seeding every reset would make all episodes identical; np.random.seed does not reach the env.)
            if seed is not None and ep == 0:
                state = env.reset(seed=seed)
            else:
                state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            done = False
            steps = 0
            episode_reward = 0.0
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
                episode_reward += reward
            # Classify how the episode ended (sparse reward: only the final step can be positive)
            if terminated and reward > 0:
                outcome = 'success'
            elif terminated:
                outcome = 'fail'
            else:
                outcome = 'timeout'
            outcomes[outcome] += 1
            raw_steps = steps
            # A failed or timed-out episode counts as the environment's step limit
            if max_steps is not None and outcome != 'success':
                steps = max_steps
            total_steps += steps
            all_steps.append(steps)
            episodes.append((raw_steps, steps, episode_reward, outcome))

        return total_steps / n_episodes, total_reward / n_episodes, all_steps, outcomes, episodes
    except Exception as e:
        print(f"Worker Error: {e}")
        import traceback
        traceback.print_exc()
        worst = float(max_steps) if max_steps is not None else 200.0
        return (worst, 0.0, [worst]*n_episodes, {'success': 0, 'fail': 0, 'timeout': 0, 'error': n_episodes},
                [(None, worst, 0.0, 'error')]*n_episodes)


def _fmt_outcomes(outcomes):
    """Short text like 'Success=48% (fail 47, timeout 5)' for the logs."""
    n = sum(outcomes.values())
    rate = 100.0 * outcomes['success'] / n if n else 0.0
    text = f"Success={rate:.0f}% (fail {outcomes['fail']}, timeout {outcomes['timeout']}"
    if outcomes['error']:
        text += f", error {outcomes['error']}"
    return text + ")"

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

    # Every evaluation (call to _run_episodes) gets its own seed derived from the run seed, so the
    # whole run is reproducible while each candidate still plays different episodes.
    # Without --seed, episodes stay random as before.
    eval_counter = itertools.count()
    def eval_seed():
        return None if args.seed is None else args.seed * 1_000_000 + next(eval_counter)

    suffix = f"_run{args.run_id}" if args.run_id else ""

    WEIGHT_FILE, ENV_CLS, N_INPUT, N_OUTPUT, CONTINUOUS, MAX_STEPS = ENV_CONFIGS[ENV_NAME]
    print(f"Environment: {ENV_NAME}  (inputs={N_INPUT}, outputs={N_OUTPUT}, continuous={CONTINUOUS}){f' | Run: {args.run_id}' if args.run_id else ''}")
    
    POP_SIZE = 16         # 16 pairs = 32 individuals
    GENERATIONS = 100
    LEARNING_RATE = 0.05
    SIGMA = 0.05          
    N_EVALS_PER_CAND = 100
    
    print("--- 0. Shared Weight Baseline ---")
    w_vals = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
    best_shared_steps = float('inf')  # any first result is better (works for every environment's step limit)
    best_shared_weight = None
    best_shared_steps_list = None
    # Every 200-episode evaluation of this run, saved at the end in the summary/episodes CSVs
    final_evaluations = []  # (config, shared_weight, avg_steps, avg_reward, steps_list, outcomes, episodes)

    for w in w_vals:
        avg_steps, avg_reward, steps_list, outcomes, episodes = _run_episodes(n_episodes=200, env_cls=ENV_CLS, continuous=CONTINUOUS, n_input=N_INPUT, n_output=N_OUTPUT, weight_file=WEIGHT_FILE, shared_weight=w, trainable=False, max_steps=MAX_STEPS, seed=eval_seed())
        std_dev = np.std(steps_list)
        print(f"Weight {w: .1f}: Avg Steps={avg_steps:.1f} ± {std_dev:.1f}, Avg Reward={avg_reward:.1f}, {_fmt_outcomes(outcomes)}")
        final_evaluations.append((f"baseline_w{w}", w, avg_steps, avg_reward, steps_list, outcomes, episodes))
        if avg_steps < best_shared_steps:
            best_shared_steps = avg_steps
            best_shared_weight = w
            best_shared_steps_list = steps_list
            best_shared_outcomes = outcomes
    print(f"Best Shared Weight Baseline: {best_shared_weight:.1f} with {best_shared_steps:.1f} steps.\n")

    print("--- 1. ES Training ---")

    # Create a master tmp model to track weights update
    tmp_model = WannPyTorch(WEIGHT_FILE, n_input=N_INPUT, n_output=N_OUTPUT, trainable=True)
    nn.init.constant_(tmp_model.edge_weights, best_shared_weight)
    current_weights = tmp_model.edge_weights.detach().cpu().numpy()
    # best_overall tracks the best master policy weights (evolved via ES) evaluated across all generations
    best_overall_steps = float('inf')
    best_overall_weights = current_weights.copy()

    # best_mutant tracks the best individual mutant candidate evaluated across all generations
    best_mutant_steps = float('inf')
    best_mutant_weights = current_weights.copy()

    # History tracking for plots
    history_avg_steps = []
    history_avg_reward = []
    history_eval_steps = []
    history_eval_reward = []
    history_best_candidate_steps = []
    history_best_overall_steps = []
    history_eval_success = []   # success rate of the updated policy (fraction of the 100 episodes)
    history_no_signal = []      # 1 if all candidates had the same reward in that generation

    # Number of generations in which all candidates had the same reward (no ES signal)
    no_signal_generations = 0

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
                                n_input=N_INPUT, n_output=N_OUTPUT, weight_file=WEIGHT_FILE, edge_weights_np=cand, trainable=True,
                                max_steps=MAX_STEPS, seed=eval_seed()))
            
            # OpenAI ES Update Logic (Vectorized)
            # For each candidate extract the average step count
            results_np_s = np.array([f.result()[0] for f in futures])
            results_np_r = np.array([f.result()[1] for f in futures])

            # Warn when ES has no signal: if every candidate has the same reward (e.g. all 0),
            # the ranking is arbitrary and this generation's update is random.
            no_signal = bool(np.all(results_np_r == results_np_r[0]))
            if no_signal:
                no_signal_generations += 1
                print(f"WARNING: all {len(results_np_r)} candidates have the same reward ({results_np_r[0]:.1f}): "
                      f"no ES signal in this generation, the update is random.")

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
                            edge_weights_np=current_weights.copy(), trainable=True, max_steps=MAX_STEPS, seed=eval_seed())
            eval_steps, eval_reward, _, eval_outcomes, _ = eval_future.result()

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
            history_eval_success.append(eval_outcomes['success'] / max(1, sum(eval_outcomes.values())))
            history_no_signal.append(int(no_signal))

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
                
            print(f"Gen {gen + 1:03d}/{GENERATIONS} | Pop Avg: {gen_avg:.1f} | Updated Policy: {eval_steps:.1f} ({_fmt_outcomes(eval_outcomes)}) | Best Policy: {best_overall_steps:.1f} | Best Mutant: {best_mutant_steps:.1f} | ETA: {eta_str} (Time: {duration:.2f}s)")
            
            if best_overall_steps < 40: 
                print("Task solved! Stopping training.")
                break

    print(f"\nGenerations without ES signal: {no_signal_generations}/{len(history_eval_steps)}")

    print("\n--- 2. Final Trained Network Verification ---")
    
    # 1. Evaluate the best policy weights (best_overall_weights) over 200 episodes
    print("Evaluating final policy weights...")
    final_policy_steps, final_policy_reward, final_policy_list, final_policy_outcomes, final_policy_episodes = _run_episodes(
        200, env_cls=ENV_CLS, continuous=CONTINUOUS, weight_file=WEIGHT_FILE,
        n_input=N_INPUT, n_output=N_OUTPUT, edge_weights_np=best_overall_weights, max_steps=MAX_STEPS,
        seed=eval_seed()
    )
    
    # 2. Evaluate the best mutant weights (best_mutant_weights) over 200 episodes
    print("Evaluating best mutant weights...")
    final_mutant_steps, final_mutant_reward, final_mutant_list, final_mutant_outcomes, final_mutant_episodes = _run_episodes(
        200, env_cls=ENV_CLS, continuous=CONTINUOUS, weight_file=WEIGHT_FILE,
        n_input=N_INPUT, n_output=N_OUTPUT, edge_weights_np=best_mutant_weights, max_steps=MAX_STEPS,
        seed=eval_seed()
    )
    
    std_base = np.std(best_shared_steps_list)
    std_policy = np.std(final_policy_list)
    std_mutant = np.std(final_mutant_list)
    
    # Welch's t-test comparisons
    t_stat_p, p_value_p = stats.ttest_ind(best_shared_steps_list, final_policy_list, equal_var=False)
    t_stat_m, p_value_m = stats.ttest_ind(best_shared_steps_list, final_mutant_list, equal_var=False)

    print("\n==================== FINAL COMPARISON ====================")
    print(f"Best Shared Baseline (200 episodes):        Avg Steps = {best_shared_steps:.1f} ± {std_base:.1f}, {_fmt_outcomes(best_shared_outcomes)}")
    print(f"Best Mutant (200 episodes):     Avg Steps = {final_mutant_steps:.1f} ± {std_mutant:.1f}, Avg Reward = {final_mutant_reward:.1f}, {_fmt_outcomes(final_mutant_outcomes)}")
    print(f"Best Policy (200 episodes):     Avg Steps = {final_policy_steps:.1f} ± {std_policy:.1f}, Avg Reward = {final_policy_reward:.1f}, {_fmt_outcomes(final_policy_outcomes)}")
    print("----------------------------------------------------------")
    print(f"Welch's t-test p-value (Baseline vs Mutant): {p_value_m:.6f}")
    print(f"Welch's t-test p-value (Baseline vs Policy): {p_value_p:.6f}")
    print("==========================================================\n")
    
    best_final_steps = min(final_policy_steps, final_mutant_steps)
    if best_final_steps < best_shared_steps:
        print(f"Success! Performance improved from {best_shared_steps:.1f} to {best_final_steps:.1f} steps.")
    else:
        print("Final verification did not show improvement over baseline.")

    # Save all run data BEFORE plotting, so nothing is lost if a plot fails.
    # These files make it possible to rebuild tables and plots, or re-evaluate the weights, without re-running ES.
    plot_dir = "plots"
    final_evaluations.append(("best_policy", None, final_policy_steps, final_policy_reward,
                              final_policy_list, final_policy_outcomes, final_policy_episodes))
    final_evaluations.append(("best_mutant", None, final_mutant_steps, final_mutant_reward,
                              final_mutant_list, final_mutant_outcomes, final_mutant_episodes))
    run_label = args.run_id if args.run_id else ""
    data_csv_dir = os.path.join(plot_dir, 'csv')
    os.makedirs(data_csv_dir, exist_ok=True)

    # 1. Summary: one row per 200-episode evaluation (6 shared weights, best policy, best mutant)
    summary_csv_path = os.path.join(data_csv_dir, f'{ENV_NAME}_summary{suffix}.csv')
    with open(summary_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Env', 'Run', 'Seed', 'Config', 'Shared_Weight', 'Episodes', 'Avg_Steps', 'Std_Steps',
                         'Avg_Reward', 'Success', 'Fail', 'Timeout', 'Error', 'Success_Rate',
                         'Best_Shared_Weight', 'Generations_Run', 'Generations_Without_Signal', 'Max_Steps'])
        for config, w, avg_s, avg_r, s_list, outc, _ in final_evaluations:
            n_ep = sum(outc.values())
            writer.writerow([ENV_NAME, run_label, args.seed, config, '' if w is None else w, n_ep,
                             avg_s, np.std(s_list), avg_r, outc['success'], outc['fail'], outc['timeout'],
                             outc['error'], outc['success'] / max(1, n_ep),
                             best_shared_weight, len(history_eval_steps), no_signal_generations, MAX_STEPS])
    print(f"Saved: {summary_csv_path}")

    # 2. Episodes: every single episode of those evaluations
    episodes_csv_path = os.path.join(data_csv_dir, f'{ENV_NAME}_episodes{suffix}.csv')
    with open(episodes_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Config', 'Episode', 'Raw_Steps', 'Counted_Steps', 'Reward', 'Outcome'])
        for config, _, _, _, _, _, eps in final_evaluations:
            for i, (raw_s, counted_s, ep_r, outcome) in enumerate(eps):
                writer.writerow([config, i, '' if raw_s is None else raw_s, counted_s, ep_r, outcome])
    print(f"Saved: {episodes_csv_path}")

    # 3. Generations: all per-generation metrics of the ES training
    generations_csv_path = os.path.join(data_csv_dir, f'{ENV_NAME}_generations{suffix}.csv')
    with open(generations_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Generation', 'Policy_Eval_Steps', 'Policy_Eval_Reward', 'Policy_Eval_Success_Rate',
                         'Pop_Avg_Steps', 'Pop_Avg_Reward', 'Best_Candidate_Steps', 'Best_Overall_Steps',
                         'No_ES_Signal', 'Best_Baseline_Steps'])
        for g in range(len(history_eval_steps)):
            writer.writerow([g, history_eval_steps[g], history_eval_reward[g], history_eval_success[g],
                             history_avg_steps[g], history_avg_reward[g], history_best_candidate_steps[g],
                             history_best_overall_steps[g], history_no_signal[g], best_shared_steps])
    print(f"Saved: {generations_csv_path}")

    # 4. Final weights, to re-evaluate them later without re-running ES
    weights_dir = os.path.join(plot_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, f'{ENV_NAME}{suffix}.npz')
    np.savez(weights_path, best_policy=best_overall_weights, best_mutant=best_mutant_weights,
             best_shared_weight=best_shared_weight, adjacency=tmp_model.adjacency.cpu().numpy(),
             weight_file=WEIGHT_FILE, seed=-1 if args.seed is None else args.seed)
    print(f"Saved: {weights_path}")

    # Plot Training Curves
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
    # lunar_lander: automatic y-axis (values now range up to the 1000-step limit)
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
    # lunar_lander: automatic y-axis (values now range up to the 1000-step limit)
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
