import numpy as np

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
    - query: Query tensor of shape (batch_size, num_heads, seq_length_q, key_dim).
    - key: Key tensor of shape (batch_size, num_heads, seq_length_k, key_dim).
    - value: Value tensor of shape (batch_size, num_heads, seq_length_v, value_dim).
    - mask: Optional mask tensor of shape (batch_size, seq_length_q, seq_length_k) or broadcastable shape.

    Returns:
    - attention_output: Output tensor after applying attention mechanism of shape (batch_size, num_heads, seq_length_q, value_dim).
    - attention_weights: Attention weights tensor of shape (batch_size, num_heads, seq_length_q, seq_length_k).
    """

    key_dim = key.shape[-1]
    scaled_scores = np.matmul(query, key.transpose((0, 1, 3, 2))) / np.sqrt(key_dim)

    if mask is not None:
        scaled_scores = scaled_scores + (mask * -1e9)

    attention_weights = np.softmax(scaled_scores, axis=-1)
    attention_output = np.matmul(attention_weights, value)

    return attention_output, attention_weights


def self_attention(input_embeddings, query_dim, key_dim, value_dim, num_heads, mask=None):
    """
    Apply self-attention mechanism.

    Args:
    - input_embeddings: Input embeddings of shape (batch_size, seq_length, embedding_dim).
    - query_dim: Dimensionality of the Query vector.
    - key_dim: Dimensionality of the Key vector.
    - value_dim: Dimensionality of the Value vector.
    - num_heads: Number of attention heads.
    - mask: Optional mask tensor of shape (batch_size, seq_length, seq_length) or broadcastable shape.

    Returns:
    - attention_output: Output tensor after applying self-attention mechanism of shape (batch_size, seq_length, value_dim).
    - attention_weights: Attention weights tensor of shape (batch_size, num_heads, seq_length, seq_length).
    """

    batch_size, seq_length, embedding_dim = input_embeddings.shape

    # Compute Query, Key, and Value vectors
    query = np.matmul(input_embeddings, np.random.randn(embedding_dim, query_dim))
    key = np.matmul(input_embeddings, np.random.randn(embedding_dim, key_dim))
    value = np.matmul(input_embeddings, np.random.randn(embedding_dim, value_dim))

    # Reshape Query, Key, and Value for multi-head attention
    query = query.reshape(batch_size, seq_length, num_heads, query_dim // num_heads)
    key = key.reshape(batch_size, seq_length, num_heads, key_dim // num_heads)
    value = value.reshape(batch_size, seq_length, num_heads, value_dim // num_heads)

    # Transpose to perform attention across dimensions
    query = query.transpose((0, 2, 1, 3))
    key = key.transpose((0, 2, 1, 3))
    value = value.transpose((0, 2, 1, 3))

    # Apply scaled dot-product attention for each head
    attention_output, attention_weights = scaled_dot_product_attention(query, key, value, mask)

    # Transpose and reshape back to original dimensions
    attention_output = attention_output.transpose((0, 2, 1, 3))
    attention_output = attention_output.reshape(batch_size, seq_length, value_dim)
    attention_weights = attention_weights.transpose((0, 2, 1, 3))

    return attention_output, attention_weights


# Example usage
batch_size = 2
seq_length = 5
embedding_dim = 64
input_embeddings = np.random.randn(batch_size, seq_length, embedding_dim)
query_dim = 32
key_dim = 32
value_dim = 64
num_heads = 4

attention_output, attention_weights = self_attention(input_embeddings, query_dim, key_dim, value_dim, num_heads)

