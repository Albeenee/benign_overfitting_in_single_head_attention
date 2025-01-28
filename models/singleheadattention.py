import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

range = 0.01

# DEVICE
if torch.backends.mps.is_available():
    device = torch.device("mps")

else:
        device = torch.device("cpu")
print("Using device:", device)


# SINGLE HEAD ATTENTION MODEL
class SingleHeadAttention(nn.Module):
    """
    Implements f(X; p, v) = v^T X^T softmax(X p),
    where X is (batch_size, 2, d),
          p, v are (d,).
    """
    def __init__(self, d):
        super().__init__()
        # Initialization of p and v
        self.p = nn.Parameter(torch.randn(d) * range)
        self.v = nn.Parameter(torch.randn(d) * range)

    def forward(self, X):
        # Compute the raw "logits" for attention: X p
        logits = X @ self.p  # (batch_size, 2, d) @ (d,) -> (batch_size, 2)
        
        # Apply softmax across the two tokens (dim=1)
        attn_weights = F.softmax(logits, dim=1) #(batch_size, 2)
        
        # Compute the weighted sum of the 2 tokens:
        weighted_sum = (attn_weights.unsqueeze(-1) * X).sum(dim=1) # (batch_size, d)
        
        # Dot product with v for each sample
        output = (weighted_sum * self.v).sum(dim=1) # (batch_size,)

        return output
