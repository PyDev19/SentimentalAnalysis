from torch import nn
from transformer.encoder_layer import Encoder

class Transformer(nn.Module):
    def __init__(
        self,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
        vocab_size,
        num_classes
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
            vocab_size
        )

        self.fc_out = nn.Linear(embed_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        enc_out = self.encoder(x, mask)
        out = self.fc_out(enc_out[:, 0, :])
        return out
