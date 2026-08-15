import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.pallas as pallas
from jax.interpreters import pallas as pl
import jax.numpy as jnp

def workload(hidden, weight, labels):
    """Fused linear + cross-entropy loss kernel for TPU."""
    batch_size, hidden_dim = hidden.shape
    _, vocab_size = weight.shape
    
    # Block size for batch dimension - use multiple of 8 for bf16
    block_batch = 128
    # Block size for vocab dimension - use multiple of 128 for vectorization
    block_vocab = 128
    
    # Grid: process batch in blocks
    grid = (batch_size // block_batch,)
    
    def cross_entropy_kernel(ref_hidden, ref_weight, ref_labels, ref_logits):
        """Kernel that computes logits and cross-entropy loss."""
        # Get program ID for batch offset
        batch_offset = pl.program_id(0) * block_batch
        
        # Compute logits for this block: (block_batch, hidden_dim) @ (hidden_dim, vocab_size) -> (block_batch, vocab_size)
        # We need to compute the full logits for this batch block
        hidden_block = ref_hidden[batch_offset:batch_offset + block_batch, :]
        
        # Compute logits using dot product
        logits_block = jnp.dot(hidden_block, ref_weight)
        
        # Compute log_softmax along vocab axis
        log_probs = jax.nn.log_softmax(logits_block, axis=-1)
        
        # Get labels for this batch block
        labels_block = ref_labels[batch_offset:batch_offset + block_batch]
        
        # Gather log probabilities for the correct labels
        # Use one-hot encoding approach
        one_hot = jax.nn.one_hot(labels_block, vocab_size)
        
        # Compute cross-entropy: -sum(one_hot * log_probs, axis=-1)
        loss_per_sample = -jnp.sum(one_hot * log_probs, axis=-1)
        
        # Return the mean loss for this block
        return jnp.mean(loss_per_sample)
    
    # Output shape is scalar
    out_shape = jax.ShapeDtypeStruct((), jnp.float32)
    
    return pl.pallas_call(
        cross_entropy_kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=(
            pl.BlockSpec((block_batch, hidden_dim), lambda i: (i * block_batch, 0)),
            pl.BlockSpec((hidden_dim, vocab_size), lambda i: (0, 0)),
            pl.BlockSpec((block_batch,), lambda i: (i * block_batch,)),
        ),
        out_specs=pl.BlockSpec((), lambda i: ()),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
    )(hidden, weight, labels)
