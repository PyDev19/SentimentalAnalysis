import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, dimensions, heads):
        super(MultiHeadAttention, self).__init__()
        assert dimensions % heads == 0, "dimensions must be divisible by heads"

        self.dimensions = dimensions # get the dimensions
        self.heads = heads # get the number of heads
        self.head_dim = dimensions // heads # get the dimension of each head
        
        self.query = nn.Linear(dimensions, dimensions) # linear layer for query
        self.key = nn.Linear(dimensions, dimensions) # linear layer for key
        self.value = nn.Linear(dimensions, dimensions) # linear layer for value

        self.output = nn.Linear(dimensions, dimensions) # linear layer for output
        
    
    def scaled_dot_product_attention(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(self.head_dim) # matrix multiplication of query and key to get attention scores

        if mask is not None: # if mask is not None, fill the scores with -inf
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(scores, dim=-1) # apply softmax to scores to get attention probabilities
        output = torch.matmul(attention, value) # matrix multiplication of attention and value to get output
        return output
    
    def split_heads(self, x):
        batch_size, seq_length, dimensions = x.size() # get the batch size, sequence length, and model dimensions of x to split the heads
        return x.view(batch_size, seq_length, self.heads, self.head_dim).transpose(1, 2) # split the heads and transpose the dimensions

    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.dimensions) # transpose the dimensions and reshape the tensor

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.query(Q)) # apply linear layer to query and split the heads
        K = self.split_heads(self.key(K)) # apply linear layer to key and split the heads
        V = self.split_heads(self.value(V)) # apply linear layer to value and split the heads
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask) # apply scaled dot product attention
        
        output = self.output(self.combine_heads(attn_output)) # combine the heads and apply linear layer to get output
        return output