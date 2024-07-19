import torch
from torch import nn, Tensor

class FeedForward(nn.Module):
    def __init__(self, dimensions: int, hidden_dim: int):
        """
        Initialize the FeedForward module.

        Args:
            dimensions (int): The input and output dimensions of the module.
            hidden_dim (int): The dimension of the intermediate hidden layer.
        """
        super(FeedForward, self).__init__()

        self.linear_1 = nn.Linear(dimensions, hidden_dim) # linear layer for the first layer
        self.relu = nn.ReLU() # ReLU activation function
        self.linear_2 = nn.Linear(hidden_dim, dimensions) # linear layer for the second layer
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass of the feedforward network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.linear_1(x) # linear transformation
        x = self.relu(x) # ReLU activation
        x = self.linear_2(x) # linear transformation
        return x