"""
Analyze memory allocation breakdown for Qwen3 inference
"""

# Model parameters (from 0.6B model)
hidden_dim = 1024  # model dimension (embedding size per position)
vocab_size = 151936
num_layers = 28
num_heads_q = 16
num_heads_kv = 8
head_dim = 128
intermediate_dim = 2816  # MLP intermediate dimension

# Sequence info from the error
prompt_length = 3600  # approximately 14427 chars
max_new_tokens = 20

def bytes_to_gb(bytes_val):
    return bytes_val / (1024**3)

def analyze_memory():
    print("="*80)
    print("MEMORY ALLOCATION BREAKDOWN")
    print("="*80)

    # Model weights (static, loaded once)
    print("\n1. MODEL WEIGHTS (loaded from disk):")
    embedding_weights = vocab_size * hidden_dim * 4  # float32
    print(f"   Embedding: {bytes_to_gb(embedding_weights):.3f} GB")

    # Per-layer weights
    per_layer = (
        (hidden_dim * num_heads_q * head_dim) +  # Q projection
        (hidden_dim * num_heads_kv * head_dim) +  # K projection
        (hidden_dim * num_heads_kv * head_dim) +  # V projection
        (num_heads_q * head_dim * hidden_dim) +   # O projection
        (hidden_dim * intermediate_dim) +          # MLP up
        (hidden_dim * intermediate_dim) +          # MLP gate
        (intermediate_dim * hidden_dim) +          # MLP down
        (hidden_dim * 2)                           # LayerNorms
    ) * 4  # float32

    total_layer_weights = per_layer * num_layers
    print(f"   All layers (28x): {bytes_to_gb(total_layer_weights):.3f} GB")
    print(f"   TOTAL WEIGHTS: {bytes_to_gb(embedding_weights + total_layer_weights):.3f} GB")

    # Activations during inference (the problem!)
    print("\n2. ACTIVATIONS (created during inference):")

    seq_len = prompt_length  # Start with prompt length

    # Input embeddings
    embeddings = seq_len * hidden_dim * 4
    print(f"   Input embeddings ({seq_len} tokens): {bytes_to_gb(embeddings):.3f} GB")

    # Per-layer activations for ONE forward pass
    print(f"\n   PER LAYER (x{num_layers}):")

    # Attention projections
    Q = seq_len * num_heads_q * head_dim * 4
    K = seq_len * num_heads_kv * head_dim * 4
    V = seq_len * num_heads_kv * head_dim * 4
    print(f"     Q matrix: {bytes_to_gb(Q):.3f} GB")
    print(f"     K matrix: {bytes_to_gb(K):.3f} GB")
    print(f"     V matrix: {bytes_to_gb(V):.3f} GB")

    # Attention scores (this can be huge!)
    # With GQA: num_heads_q heads, each attending to seq_len positions
    attn_scores = num_heads_q * seq_len * seq_len * 4
    print(f"     Attention scores ({seq_len}x{seq_len} per head): {bytes_to_gb(attn_scores):.3f} GB")

    # Attention output
    attn_output = seq_len * num_heads_q * head_dim * 4
    print(f"     Attention output: {bytes_to_gb(attn_output):.3f} GB")

    # MLP activations
    mlp_intermediate = seq_len * intermediate_dim * 4
    print(f"     MLP intermediate: {bytes_to_gb(mlp_intermediate):.3f} GB")

    # Total per layer
    per_layer_activation = Q + K + V + attn_scores + attn_output + mlp_intermediate
    print(f"     TOTAL PER LAYER: {bytes_to_gb(per_layer_activation):.3f} GB")

    # If all layers keep activations (gradient computation style)
    all_layers_activation = per_layer_activation * num_layers
    print(f"\n   ALL LAYERS TOGETHER: {bytes_to_gb(all_layers_activation):.3f} GB")

    # The concatenation issue
    print(f"\n3. THE CONCATENATION PROBLEM:")
    print(f"   Each token, embeddings grow from {seq_len} to {seq_len+1} tokens")
    print(f"   New embeddings: {bytes_to_gb((seq_len+1) * hidden_dim * 4):.3f} GB")
    print(f"   But old embeddings ({bytes_to_gb(embeddings):.3f} GB) NOT freed immediately")
    print(f"   Peak during concat: {bytes_to_gb(embeddings + (seq_len+1) * hidden_dim * 4):.3f} GB")

    # Estimate total peak memory
    print(f"\n4. ESTIMATED PEAK MEMORY:")
    weights_on_gpu = embedding_weights + total_layer_weights

    # Assuming layers computed sequentially (best case)
    sequential_activation = embeddings + per_layer_activation

    # With all intermediate tensors
    total_estimate = weights_on_gpu + sequential_activation

    print(f"   Weights: {bytes_to_gb(weights_on_gpu):.3f} GB")
    print(f"   Activations (1 layer): {bytes_to_gb(sequential_activation):.3f} GB")
    print(f"   TOTAL: {bytes_to_gb(total_estimate):.3f} GB")

    print(f"\n5. LIKELY CULPRIT:")
    print(f"   Attention scores at seq_len={seq_len}:")
    print(f"   {num_heads_q} heads × {seq_len} × {seq_len} × 4 bytes = {bytes_to_gb(attn_scores):.3f} GB PER LAYER")
    print(f"   If tinygrad doesn't free intermediate tensors properly:")
    print(f"   {num_layers} layers × {bytes_to_gb(attn_scores):.3f} GB = {bytes_to_gb(attn_scores * num_layers):.3f} GB just for attention!")

    print(f"\n6. WHY 17.5 GB?")
    print(f"   Model weights: ~1.2 GB")
    print(f"   Embeddings: {bytes_to_gb(embeddings):.3f} GB")
    print(f"   Attention matrices across layers: ~{bytes_to_gb(attn_scores * num_layers):.3f} GB")
    print(f"   Other activations: ~{bytes_to_gb(embeddings + (Q + K + V + attn_output) * num_layers):.3f} GB")
    print(f"   TOTAL: ~{bytes_to_gb(weights_on_gpu + embeddings + (Q + K + V + attn_scores + attn_output) * num_layers):.3f} GB")

if __name__ == "__main__":
    analyze_memory()
