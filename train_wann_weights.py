
import os
import time
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
    # Configuration — choose environment: 'smc_discrete', 'smc_continuous', 'lunar_lander'
    ENV_NAME = 'smc_discrete'
    WEIGHT_FILE, ENV_CLS, N_INPUT, N_OUTPUT, CONTINUOUS = ENV_CONFIGS[ENV_NAME]
    print(f"Environment: {ENV_NAME}  (inputs={N_INPUT}, outputs={N_OUTPUT}, continuous={CONTINUOUS})")
    
    POP_SIZE = 16         # 16 pairs = 32 individuals
    GENERATIONS = 100
    LEARNING_RATE = 0.05
    SIGMA = 0.05          
    N_EVALS_PER_CAND = 10
    
    print("--- 0. Shared Weight Baseline ---")
    w_vals = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
    best_shared_steps = 200.0
    best_shared_weight = None
    best_shared_steps_list = None
    
    for w in w_vals:
        avg_steps, avg_reward, steps_list = _run_episodes(n_episodes=200, env_cls=ENV_CLS, continuous=CONTINUOUS, n_input=N_INPUT, n_output=N_OUTPUT, weight_file=WEIGHT_FILE, shared_weight=w, trainable=False)
        print(f"Weight {w: .1f}: Avg Steps={avg_steps:.1f}, Avg Reward={avg_reward:.1f}")
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
    best_overall_steps = 200.0
    best_overall_weights = current_weights.copy()

    # History tracking for plots
    history_avg_steps = []
    history_avg_reward = []
    history_eval_steps = []
    history_eval_reward = []

    
    print(f"Starting Advanced ES Training on {os.cpu_count()} cores...")

    # Run parallel tasks
    with ProcessPoolExecutor() as executor:
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
            if gen_min < best_overall_steps:
                best_overall_steps = gen_min
                best_idx = np.argmin(results_np_s)
                best_overall_weights = candidates[best_idx].copy()
            
            # Evaluate current policy with updated weights (100 episodes)
            eval_future = executor.submit(_run_episodes, 100, env_cls=ENV_CLS, continuous=CONTINUOUS,
                            n_input=N_INPUT, n_output=N_OUTPUT, weight_file=WEIGHT_FILE,
                            edge_weights_np=current_weights.copy(), trainable=True)
            eval_steps, eval_reward, _ = eval_future.result()
            
            # Track history
            history_avg_steps.append(gen_avg)
            history_avg_reward.append(gen_avg_reward)
            history_eval_steps.append(eval_steps)
            history_eval_reward.append(eval_reward)
            
            duration = time.time() - start_time
            print(f"Gen {gen:03d} | Pop Avg: {gen_avg:.1f} | Updated Policy: {eval_steps:.1f} | Best Overall: {best_overall_steps:.1f} | Time: {duration:.2f}s")
            
            if best_overall_steps < 40: 
                print("Task solved! Stopping training.")
                break

    print("\n--- 2. Final Trained Network Verification ---")
    final_avg_steps, final_avg_reward, final_steps_list = _run_episodes(200, env_cls=ENV_CLS, continuous=CONTINUOUS, weight_file=WEIGHT_FILE, n_input=N_INPUT, n_output=N_OUTPUT, edge_weights_np=best_overall_weights)
    
    std_base = np.std(best_shared_steps_list)
    std_es = np.std(final_steps_list)
    t_stat, p_value = stats.ttest_ind(best_shared_steps_list, final_steps_list, equal_var=False)

    print(f"Best Shared Baseline (200 episodes): Avg Steps = {best_shared_steps:.1f} ± {std_base:.1f}")
    print(f"Final Trained Performance (200 episodes): Avg Steps = {final_avg_steps:.1f} ± {std_es:.1f}, Avg Reward = {final_avg_reward:.1f}")
    print(f"Welch's t-test p-value: {p_value:.6f}")
    
    if final_avg_steps < best_shared_steps:
        print(f"Success! Performance improved from {best_shared_steps:.1f} to {final_avg_steps:.1f} steps.")
    else:
        print("Final verification did not show improvement over baseline.")

    # Plot Training Curves
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    generations = list(range(len(history_avg_steps)))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(generations, history_eval_steps, linewidth=2, color='#2196F3', label='Policy Eval Steps (100 runs)')
    ax.axhline(y=best_shared_steps, linestyle='--', linewidth=2, color='#E53935', label=f'Baseline (shared w={best_shared_weight:.1f})')
    ax.set_xlabel('Generation', fontsize=14)
    ax.set_ylabel('Average Steps', fontsize=14)
    ax.set_title('ES Weight Training — Steps per Generation', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'{ENV_NAME}_training.pdf'), format='pdf')
    plt.close(fig)
    print(f"Saved: {plot_dir}/{ENV_NAME}_training.pdf")

    
if __name__ == "__main__":
    main()
