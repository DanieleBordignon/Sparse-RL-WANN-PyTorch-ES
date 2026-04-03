
import torch
import numpy as np
from convert_wann import WannPyTorch
from wann_src.ind import importNet, act

def compare_implementations():
    # Use the existing smc_best.out if it exists
    weight_file = 'champions/smc_best.out'
    n_input = 2
    n_output = 3
    weight_val = 1.0
    
    # Dummy input
    dummy_input_np = np.array([[0.5, -0.2]])
    dummy_input_torch = torch.from_numpy(dummy_input_np).float()
    
    # 1. Original NumPy implementation
    wVec, aVec, _ = importNet(weight_file)
    
    dim = int(np.sqrt(len(wVec)))
    wMat = np.reshape(wVec, (dim, dim))
    wMat[np.isnan(wMat)] = 0
    wMat[wMat != 0] = 1.0
    wMat_shared = wMat * weight_val
    
    output_np = act(wMat_shared, aVec, n_input, n_output, dummy_input_np)
    
    # 2. PyTorch implementation
    model = WannPyTorch(weight_file, n_input, n_output, trainable=False)
    output_torch = model(dummy_input_torch, weight=weight_val).detach().numpy()
    
    print(f"Original NumPy Output: {output_np}")
    print(f"PyTorch implementation Output: {output_torch}")
    
    diff = np.abs(output_np - output_torch)
    print(f"Max Difference: {np.max(diff)}")
    
    if np.allclose(output_np, output_torch, atol=1e-6):
        print("SUCCESS: Implementations are numerically identical!")
    else:
        print("FAILURE: Implementations differ!")

if __name__ == "__main__":
    compare_implementations()
