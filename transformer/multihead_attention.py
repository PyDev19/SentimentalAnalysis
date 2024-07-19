import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, dimensions, heads):
        """
        Initialize the MultiHeadAttention module.

        Args:
            dimensions (int): The input and output dimensions of the module.
            heads (int): The number of heads to split the input into.
        """
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
        """
        Compute scaled dot product attention.

        Args:
            query (torch.Tensor): The query tensor of shape (batch_size, num_heads, query_length, head_dim).
            key (torch.Tensor): The key tensor of shape (batch_size, num_heads, key_length, head_dim).
            value (torch.Tensor): The value tensor of shape (batch_size, num_heads, value_length, head_dim).
            mask (torch.Tensor, optional): The mask tensor of shape (batch_size, num_heads, query_length, key_length).
                If provided, the attention scores will be masked with -inf where the mask is 0.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_heads, query_length, head_dim).

        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(self.head_dim) # matrix multiplication of query and key to get attention scores

        if mask is not None: # if mask is not None, fill the scores with -inf
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(scores, dim=-1) # apply softmax to scores to get attention probabilities
        output = torch.matmul(attention, value) # matrix multiplication of attention and value to get output
        return output
    
    def split_heads(self, x):
        """
        Split the input tensor into multiple heads.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_length, dimensions).

        Returns:
            torch.Tensor: The tensor with dimensions split into multiple heads and transposed.

        """
        batch_size, seq_length, dimensions = x.size() # get the batch size, sequence length, and model dimensions of x to split the heads
        return x.view(batch_size, seq_length, self.heads, self.head_dim).transpose(1, 2) # split the heads and transpose the dimensions

    def combine_heads(self, x):
        """
        Combines the multiple heads back to the original shape.

        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, num_heads, seq_length, dimensions).

        Returns:
            torch.Tensor: The combined tensor with shape (batch_size, seq_length, dimensions).
        """
        batch_size, _, seq_length, dimensions = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.dimensions)

    def forward(self, Q, K, V, mask=None):
        """
        Perform forward pass of the multi-head attention layer.

        Args:
            Q (torch.Tensor): The query tensor.
            K (torch.Tensor): The key tensor.
            V (torch.Tensor): The value tensor.
            mask (torch.Tensor, optional): The attention mask tensor. Defaults to None.

        Returns:
            torch.Tensor: The output tensor after applying multi-head attention.

        """
        Q = self.split_heads(self.query(Q)) # apply linear layer to query and split the heads
        K = self.split_heads(self.key(K)) # apply linear layer to key and split the heads
        V = self.split_heads(self.value(V)) # apply linear layer to value and split the heads

        attention_output = self.scaled_dot_product_attention(Q, K, V, mask) # apply scaled dot product attention

        output = self.output(self.combine_heads(attention_output)) # combine the heads and apply linear layer to get output
        return output