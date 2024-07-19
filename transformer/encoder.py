from torch import nn, Tensor
from multihead_attention import MultiHeadAttention
from feedforward import FeedForward

class Encoder(nn.Module):
    def __init__(self, dimensions: int, heads: int, hidden_dim: int, dropout: float):
        """
        Initializes the Encoder module of the Transformer model.

        Args:
            dimensions (int): The input and output dimensions of the encoder.
            heads (int): The number of attention heads in the multihead attention layer.
            hidden_dim (int): The dimension of the hidden layer in the feedforward layer.
            dropout (float): The dropout probability.

        """
        super(Encoder, self).__init__()

        self.dropout = nn.Dropout(dropout) # dropout layer

        self.multihead_attention = MultiHeadAttention(dimensions, heads) # multihead attention layer
        self.norm_1 = nn.LayerNorm(dimensions) # first normalization layer
        self.feedforward = FeedForward(dimensions, hidden_dim) # feedforward layer
        self.norm_2 = nn.LayerNorm(dimensions) # second normalization layer

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass of the encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, hidden_size).
            mask (torch.Tensor): Mask tensor of shape (batch_size, seq_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, hidden_size).
        """
        attention = self.multihead_attention(x, x, x, mask) # get the attention output from the multihead attention layer
        norm_1 = self.norm_1(x + self.dropout(attention)) # add the attention output to the input tensor and normalize

        feedforward = self.feedforward(norm_1) # get the output from the feedforward layer
        norm_2 = self.norm_2(x + self.dropout(feedforward)) # add the feedforward output to the input tensor and normalize

        return norm_2 # return the output tensor