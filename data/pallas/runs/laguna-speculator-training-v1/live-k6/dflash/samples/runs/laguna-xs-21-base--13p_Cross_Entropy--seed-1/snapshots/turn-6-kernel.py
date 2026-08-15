import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.pallas as pl
import jax.pallas.lib as pllib
import jax.pallas.tpu as pltpu

def workload(hidden, weight, labels):
    """Fused linear + cross-entropy loss for Llama 3.1 8B."""
    
    batch_size = hidden.shape[0]  # 8192
    hidden_dim = hidden.shape[1]  # 4096
    vocab_size = weight.shape[1]  # 128256
    
    # Tile sizes - use multiples of 8 for bf16 on TPU
    block_batch = 128
    block_hidden = 64
    block_vocab = 128
    
    def cross_entropy_kernel(
        hidden_ref,
        weight_ref,
        labels_ref,
        loss_ref,
        *,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
    ):
        # Grid: process batch elements in parallel
        batch_idx = pl.program_id(0)
        
        # Compute logits for this batch element: logits = hidden @ weight
        # hidden[batch_idx, :] @ weight[:, vocab_idx] for each vocab_idx
        
        # Accumulate logits in float32 for numerical stability
        logits = jnp.zeros(vocab_size, dtype=jnp.float32)
        
        # Tile the hidden dimension
        for hidden_tile in range(0, hidden_dim, block_hidden):
            hidden_end = min(hidden_tile + block_hidden, hidden_dim)
            
            # Load hidden slice
            hidden_slice = hidden_ref[batch_idx, hidden_tile:hidden_end]
            
            # Compute partial logits
            # weight_ref[:, vocab_idx] for all vocab_idx
            for vocab_tile in range(0, vocab_size, block_vocab):
                vocab_end = min(vocab_tile + block_vocab, vocab_size)
                
                # Load weight slice
                weight_slice = weight_ref[hidden_tile:hidden_end, vocab_tile:vocab_end]
                
                # Compute partial logits
                partial_logits = jnp.dot(hidden_slice.astype(jnp.float32), weight_slice.astype(jnp.float32))
                
                # Accumulate
                logits = logits.at[vocab_tile:vocab_end].add(partial_logits)
        
        # Compute log_softmax
        max_logit = jnp.max(logits)
        log_sum_exp = jnp.sum(jnp.exp(logits - max_logit))
        log_probs = logits - max_logit - jnp.log(log_sum_exp)
        
        # Get the label for this batch element
        label = labels_ref[batch_idx]
        
        # Get the log probability for the correct class
        correct_log_prob = log_probs[label]
        
        # Store the negative log probability (loss for this sample)
        loss_ref[...] = -correct_log_prob
    
    # Grid spec for batch parallelism
    grid = (batch_size,)
    
    # Block specs
    hidden_spec = pl.BlockSpec((batch_size, hidden_dim), lambda batch_idx: (batch_idx, 0))
    weight_spec = pl.BlockSpec((hidden_dim, vocab_size), lambda: (0, 0))
    labels_spec = pl.BlockSpec((batch_size,), lambda batch_idx: (batch_idx,))
    loss_spec = pl.BlockSpec((), lambda: ())
    
    # Output shape
    out_shape = jax.ShapeDtypeStruct((), jnp.float32)
    
    # Call the kernel
    losses = pl.pallas_call(
        cross_entropy_kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size,), jnp.float32),
        grid=grid,
        in_specs=(hidden_spec, weight_spec, labels_spec),
        out_specs=(loss_spec,),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
    )(hidden, weight, labels)
    
    # Return mean loss
    return jnp.mean(losses)
