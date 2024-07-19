from torch import nn, Tensor
from transformer.multihead_attention import MultiHeadAttention
from transformer.feedforward import FeedForward

class Decoder(nn.Module):
    def __init__(self, dimensions: int, heads: int, hidden_dim: int, dropout: float):
        """
        Initializes a Decoder object.

        Args:
            dimensions (int): The dimensionality of the input and output tensors.
            heads (int): The number of attention heads.
            hidden_dim (int): The dimensionality of the hidden layer in the feedforward network.
            dropout (float): The dropout probability.

        Returns:
            None
        """
        super(Decoder, self).__init__()

        self.dropout = nn.Dropout(dropout) # dropout layer

        self.masked_attention = MultiHeadAttention(dimensions, heads) # multihead attention layer
        self.norm_1 = nn.LayerNorm(dimensions) # first normalization layer

        self.cross_attention = MultiHeadAttention(dimensions, heads) # cross multihead attention layer
        self.norm_2 = nn.LayerNorm(dimensions) # second normalization layer

        self.feedforward = FeedForward(dimensions, hidden_dim) # feedforward layer
        self.norm_3 = nn.LayerNorm(dimensions) # third normalization layer
    
    def forward(self, x: Tensor, encoder_output: Tensor, source_mask: Tensor, target_mask: Tensor) -> Tensor:
        """
        Forward pass of the decoder module in the transformer model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, hidden_size).
            encoder_output (Tensor): Output tensor from the encoder module of shape (batch_size, seq_length, hidden_size).
            source_mask (Tensor): Mask tensor for the encoder output of shape (batch_size, seq_length).
            target_mask (Tensor): Mask tensor for the decoder input of shape (batch_size, seq_length).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_length, hidden_size).
        """
        masked_attention = self.masked_attention(x, x, x, target_mask) # get the masked attention output
        norm_1 = self.norm_1(x + self.dropout(masked_attention)) # add the masked attention output to the input tensor and normalize

        cross_attention = self.cross_attention(norm_1, encoder_output, encoder_output, source_mask) # get the cross attention output
        norm_2 = self.norm_2(norm_1 + self.dropout(cross_attention)) # add the cross attention output to the input tensor and normalize

        feedforward = self.feedforward(norm_2) # get the output from the feedforward layer
        norm_3 = self.norm_3(norm_2 + self.dropout(feedforward)) # add the feedforward output to the input tensor and normalize

        return norm_3 # return the output tensor