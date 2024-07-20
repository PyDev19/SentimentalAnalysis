from typing import Tuple
import torch
from torch import nn, Tensor
from transformer.encoder import Encoder
from transformer.positional_encoding import PositionalEncoding

class TransformerForSentimentAnalysis(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dimensions: int,
        heads: int,
        layers: int,
        hidden_dimensions: int,
        max_seq_length: int,
        num_classes: int,
        dropout: float,
        device: torch.device
    ):
        """
        Initializes a Transformer model for sentiment analysis.

        Args:
            vocab_size (int): The size of the vocabulary.
            dimensions (int): The dimensionality of the input and output embeddings.
            heads (int): The number of attention heads.
            layers (int): The number of encoder layers.
            hidden_dimensions (int): The dimensionality of the hidden layer in the feed-forward network.
            max_seq_length (int): The maximum sequence length.
            num_classes (int): The number of output classes (sentiments).
            dropout (float): The dropout probability.

        Returns:
            None
        """
        super(TransformerForSentimentAnalysis, self).__init__()

        self.embedding = nn.Embedding(vocab_size, dimensions)  # embedding layer
        self.positional_encoding = PositionalEncoding(dimensions, max_seq_length, device)  # positional encoding layer

        self.encoder_layers = nn.ModuleList([Encoder(dimensions, heads, hidden_dimensions, dropout) for _ in range(layers)])  # list of encoder layers

        self.fc = nn.Linear(dimensions, num_classes)  # linear layer for the output
        self.dropout = nn.Dropout(dropout)  # dropout layer
        
        self.device = device  # device to run the model on
    
    def generate_mask(self, source: Tensor) -> Tensor:
        """
        Generate masks for the source tensor.

        Args:
            source (Tensor): The source tensor.

        Returns:
            Tensor: The source mask.
        """
        source_mask = (source != 0).unsqueeze(1).unsqueeze(2).to(self.device)  # create a mask for the source tensor
        return source_mask  # return the source mask

    def forward(self, source: Tensor) -> Tensor:
        """
        Forward pass of the Transformer model for sentiment analysis.

        Args:
            source (Tensor): Input source tensor.

        Returns:
            Tensor: Output tensor after passing through the Transformer model.
        """
        source = source.to(self.device)  # ensure source is on the correct device
        source_mask = self.generate_mask(source)  # generate the source mask

        source_embedded = self.dropout(self.positional_encoding(self.embedding(source)))  # apply dropout, positional encoding, and embedding to the source tensor

        encoder_output = source_embedded  # set the encoder output to the source tensor
        for encoder_layer in self.encoder_layers:  # iterate through the encoder layers
            encoder_output = encoder_layer(encoder_output, source_mask)  # pass the encoder and source mask to the encoder layer

        encoder_output = encoder_output.mean(dim=1)  # global average pooling
        output = self.fc(encoder_output)  # apply linear layer to the encoder output
        return output  # return the output tensor
