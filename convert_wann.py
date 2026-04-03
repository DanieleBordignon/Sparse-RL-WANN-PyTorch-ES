import torch
import torch.nn as nn
import numpy as np
import math

class WannPyTorch(nn.Module):
    def __init__(self, weight_file, n_input, n_output, trainable=False):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.trainable = trainable
        
        # Load data
        # Data format: [N x (N+1)] matrix
        # Columns 0..N-1: Adjacency Matrix (NxN) - 0 or Nan: No connection, !=0: Connection
        # Column N: Activation Function IDs
        try:
            data = np.loadtxt(weight_file, delimiter=',')
        except OSError:
            raise FileNotFoundError(f"Could not find weight file at {weight_file}")
            
        # Separate adjacency and activations
        self.adj_mat_np = data[:, :-1]  # Matrix of shape (N, N)
        self.a_vec_np = data[:, -1]   # Vector of shape (N,)
        
        # Replace NaNs with 0
        self.adj_mat_np = np.nan_to_num(self.adj_mat_np)
        
        # Ensure it is a binary mask (1.0 for connection, 0.0 for none)
        self.adj_mat_np[self.adj_mat_np != 0] = 1.0
        
        self.n_nodes = self.adj_mat_np.shape[0]
        
        # Register fixed topology buffers
        self.register_buffer('adjacency', torch.from_numpy(self.adj_mat_np).float())
        self.register_buffer('activations', torch.from_numpy(self.a_vec_np).long())
        
        # If trainable, initialize weights as parameters
        if self.trainable:
            # Initialize with random weights
            self.edge_weights = nn.Parameter(torch.randn(self.n_nodes, self.n_nodes))
        
    def forward(self, x, weight=1.0):
        """
        Forward pass node-by-node.
        x: Input tensor of shape (batch_size, n_input)
        weight: Shared weight value (ignored if self.trainable is True)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        batch_size = x.shape[0]
        
        # Determine the effective weight matrix
        if self.trainable:
            # Mask the parameters to enforce topology
            effective_weights = self.edge_weights * self.adjacency
        else:
            # Broadcast shared weight across topology
            effective_weights = self.adjacency * weight
            
        # Build the states
        # The nodes are topologically sorted in the file.
        # Node 0: Bias
        # Node 1..n_input: Inputs
        
        # Collect states as a list of tensors (each shape [batch_size, 1])
        # Start with Bias
        bias_col = torch.ones((batch_size, 1), device=x.device, dtype=x.dtype)
        state_cols = [bias_col]
        
        # Add Input nodes
        n_in_copy = min(self.n_input, self.n_nodes - 1)
        for k in range(n_in_copy):
            # Extract column k from input x
            state_cols.append(x[:, k:k+1])

        
        start_node = len(state_cols)
        
        # Process hidden/output nodes
        # From start_node (after bias and inputs) to n_nodes (last node)
        for i in range(start_node, self.n_nodes):
            # We need to compute activation for node 'i'
            # Inputs are nodes 0..i-1. 
            # Weighted sum = sum(State[j] * Weight[j->i]) for j < i
            
            # Helper to stack what we have so far
            current_states = torch.cat(state_cols, dim=1) # Shape (batch, i)
            
            # Get weights allowed to connect to i from existing nodes
            # effective_weights is (N, N). Column i is weights TO node i.
            # Row index is Source node.
            # We take the first 'i' rows of column 'i'.
            w_col = effective_weights[:i, i] # Shape (i,)
            
            # Matmul: (batch, i) x (i,) -> (batch,)
            raw_act = torch.matmul(current_states, w_col)
            
            # Apply activation
            act_id = self.activations[i].item()
            new_col = self.apply_activation(raw_act, act_id)
            
            # Reshape len (batch,) to (batch, 1) and append
            state_cols.append(new_col.unsqueeze(1))
            
        # Stack all to get full states matrix if needed, or just return outputs
        final_states = torch.cat(state_cols, dim=1)
            
        # Return the last n_output nodes
        return final_states[:, -self.n_output:]

    def apply_activation(self, x, act_id):
        # Activation mapping from WANN src/ind.py
        if act_id == 1:   # Linear
            return x
        elif act_id == 2: # Unsigned Step
            # NOTE: Step is non-differentiable. 
            # If training, you might want sigmoid or a relaxed step.
            # Keeping as-is for fidelity, but training will struggle if this is used used heavily.
            return (x > 0).float()
        elif act_id == 3: # Sin
            return torch.sin(math.pi * x)
        elif act_id == 4: # Gaussian
            return torch.exp(-0.5 * x**2)
        elif act_id == 5: # Tanh
            return torch.tanh(x)
        elif act_id == 6: # Sigmoid
            return torch.sigmoid(x)
        elif act_id == 7: # Inverse
            return -x
        elif act_id == 8: # Abs
            return torch.abs(x)
        elif act_id == 9: # Relu
            return torch.relu(x)
        elif act_id == 10: # Cosine
            return torch.cos(math.pi * x)
        elif act_id == 11: # Squared
            return x**2
        else:
            return x