from typing import Tuple
import torch
from torch import nn, Tensor
from encoder import Encoder
from decoder import Decoder
from positional_encoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        dimensions: int,
        heads: int,
        layers: int,
        hidden_dimensions: int,
        max_seq_length: int,
        dropout: float
    ):
        """
        Initializes a Transformer model.

        Args:
            source_vocab_size (int): The size of the source vocabulary.
            target_vocab_size (int): The size of the target vocabulary.
            dimensions (int): The dimensionality of the input and output embeddings.
            heads (int): The number of attention heads.
            layers (int): The number of encoder and decoder layers.
            hidden_dimensions (int): The dimensionality of the hidden layer in the feed-forward network.
            max_seq_length (int): The maximum sequence length.
            dropout (float): The dropout probability.

        Returns:
            None
        """
        super(Transformer, self).__init__()

        self.encoder_embedding = nn.Embedding(source_vocab_size, dimensions) # embedding layer for the encoder
        self.decoder_embedding = nn.Embedding(target_vocab_size, dimensions) # embedding layer for the decoder
        self.positional_encoding = PositionalEncoding(dimensions, max_seq_length) # positional encoding layer

        self.encoder_layers = nn.ModuleList([Encoder(dimensions, heads, hidden_dimensions, dropout) for _ in range(layers)]) # list of encoder layers
        self.decoder_layers = nn.ModuleList([Decoder(dimensions, heads, hidden_dimensions, dropout) for _ in range(layers)]) # list of decoder layers

        self.linear = nn.Linear(dimensions, target_vocab_size) # linear layer for the output
        self.dropout = nn.Dropout(dropout) # dropout layer
        self.softmax = nn.Softmax(dim=-1) # softmax layer
    
    def generate_mask(self, source: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Generate masks for the source and target tensors.

        Args:
            source (Tensor): The source tensor.
            target (Tensor): The target tensor.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the source mask and target mask.
        """
        source_mask = (source != 0).unsqueeze(1).unsqueeze(2) # create a mask for the source tensor
        target_mask = (target != 0).unsqueeze(1).unsqueeze(3) # create a mask for the target tensor

        seq_length = target.size(1) # get the sequence length from 2nd dimension of target tensor

        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool() # create a mask for the target tensor
        target_mask = target_mask & nopeak_mask # combine the target mask with the nopeak mask

        return source_mask, target_mask # return the source and target masks

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass of the Transformer model.

        Args:
            source (Tensor): Input source tensor.
            target (Tensor): Input target tensor.

        Returns:
            Tensor: Output tensor after passing through the Transformer model.
        """
        source_mask, target_mask = self.generate_mask(source, target) # generate the source and target masks

        source_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(source))) # apply dropout, positional encoding, and embedding to the source tensor
        target_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(target))) # apply dropout, positional encoding, and embedding to the target tensor

        encoder_output = source_embedded # set the encoder output to the source tensor
        for encoder_layer in self.encoder_layers: # iterate through the encoder layers
            encoder_output = encoder_layer(encoder_output, source_mask) # pass the encoder and source mask to the encoder layer

        decoder_output = target_embedded # set the decoder output to the target tensor
        for decoder_layer in self.decoder_layers: # iterate through the decoder layers
            decoder_output = decoder_layer(decoder_output, encoder_output, source_mask, target_mask) # pass the decoder, source, and target masks to the decoder layer

        output = self.linear(decoder_output) # apply linear layer to the decoder output
        output = self.softmax(output) # apply softmax to the output
        return output # return the output tensor