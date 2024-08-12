import numpy as np
from typing import Dict, Tuple

def forward_linear_regression(X_batch: np.ndarray, y_batch: np.ndarray,
                              weights: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, np.ndarray]]:
    assert X_batch.shape[0] == y_batch.shape[0]
    assert X_batch.shape[1] == weights['W'].shape[0]
    
    N = X_batch @ weights['W']
    P = N + weights['B']
    
    loss = np.sqrt(np.mean(np.power(y_batch - P, 2)))
    
    forward_info: Dict[str, np.ndarray] = {}
    forward_info['X'] = X_batch
    forward_info['N'] = N
    forward_info['P'] = P
    forward_info['y'] = y_batch
    
    return loss, forward_info


def loss_gradients(forward_info: Dict[str, np.ndarray], weights: Dict[str, np.ndarray])\
    -> Dict[str, np.ndarray]:
    #batch_size = forward_info['X'].shape[0]
    dLdP = -2 * (forward_info['y'] - forward_info['P'])
    
    dPdN = np.ones_like(forward_info['N'])
    dPdB = np.ones_like(weights['B'])
    
    dLdN = dLdP * dPdN
    dNdW = np.transpose(forward_info['X'], (1, 0))
    dLdW = np.dot(dNdW, dLdN)
    dLdB = (dLdP * dPdB).sum(axis=0)
    
    loss_gradients_: Dict[str, np.ndarray] = {}
    loss_gradients_['W'] = dLdW
    loss_gradients_['B'] = dLdB
    return loss_gradients_    
