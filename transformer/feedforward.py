import torch
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, dimensions, hidden_dim):
        """
        Initialize the FeedForward module.

        Args:
            dimensions (int): The input and output dimensions of the module.
            hidden_dim (int): The dimension of the intermediate hidden layer.
        """
        super(FeedForward, self).__init__()

        self.linear_1 = nn.Linear(dimensions, hidden_dim)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dim, dimensions)
    
    def forward(self, x):
        """
        Performs the forward pass of the feedforward network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x