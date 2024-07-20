import math
import torch
from torch import nn, Tensor

class PositionalEncoding(nn.Module):
    def __init__(self, dimensions: int, max_seq_length: int, device: torch.device):
        """
        Initialize the PositionalEncoding class.

        Args:
            dimensions (int): The dimensionality of the model.
            max_seq_length (int): The maximum sequence length.

        Returns:
            None
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, dimensions).to(device) # initialize the positional encoding matrix
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1).to(device) # create a tensor with values from 0 to max_seq_length
        div_term = torch.exp(torch.arange(0, dimensions, 2).float() * -(math.log(10000.0) / dimensions)).to(device) # create a tensor with values from 0 to d_model, multiplied by -log(10000) and exponentiated

        pe[:, 0::2] = torch.sin(position * div_term).to(device) # fill the even indices with the sine of the position multiplied by div_term
        pe[:, 1::2] = torch.cos(position * div_term).to(device) # fill the odd indices with the cosine of the position multiplied by div_term

        self.register_buffer('pe', pe.unsqueeze(0)) # register the positional encoding as a buffer
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The input tensor with positional encoding added.
        """
        return x + self.pe[:, :x.size(1)]  # add the positional encoding to the input tensor