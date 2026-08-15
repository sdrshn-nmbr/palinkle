import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas.tpu as pltpu

def workload(x, router_weights, expert_gate_kernels, expert_up_kernels, expert_down_kernels):
    """Sparse MoE kernel for Mixtral-8x7B.
    
    Args:
        x: [B, S, E] input tensor
        router_weights: [S, num_experts] router weights
        expert_gate_kernels: [num_experts, E, mlp_dim] gate kernels
        expert_up_kernels: [num_experts, E, mlp_dim] up kernels
        expert_down_kernels: [num_experts, mlp_dim, E] down kernels
    
    Returns:
        [B, S, E] output tensor in float32
    """
    B, S, E = x.shape
    num_experts = router_weights.shape[-1]
    num_experts_per_tok = 2  # K
    mlp_dim = expert_gate_kernels.shape[-1]
    
    def sparse_moe_kernel(
        x_ref,
        router_weights_ref,
        expert_gate_kernels_ref,
        expert_up_kernels_ref,
        expert_down_kernels_ref,
        out_ref,
    ):
        # Get batch and sequence indices
        b = pl.program_id(0)
        s = pl.program_id(1)
        
        # Compute logits = x[b, s, :] · router_weights[s, :]
        # x[b, s, :] has shape [E], router_weights[s, :] has shape [num_experts]
        # We need to compute dot product for each expert
        # logits[e] = sum over E of x[b, s, E] * router_weights[s, e]
        # Actually, looking at the AST: logits = jnp.dot(x, router_weights)
        # x is [B, S, E], router_weights is [S, num_experts]
        # jnp.dot on these would give [B, S, num_experts]
        
        # For this position, get the slice
        x_slice = x_ref[b, s, :]  # [E]
        router_slice = router_weights_ref[s, :]  # [num_experts]
        
        # Compute logits for each expert
        # logits[e] = dot(x_slice, router_weights[:, e])? No...
        # Actually jnp.dot(x, router_weights) where x is [B, S, E] and router_weights is [S, num_experts]
        # This gives [B, S, num_experts] where logits[b, s, e] = sum_E x[b, s, E] * router_weights[s, e]
        # Wait, that's not right either. Let me think again.
        
        # jnp.dot(x, router_weights) with x=[B,S,E] and router_weights=[S,num_experts]
        # The last dimension of x (E) contracts with... the first dimension of router_weights (S)?
        # No, that doesn't match.
        
        # Actually, jnp.dot on 2D arrays does matrix multiplication.
        # For 3D arrays, jnp.dot contracts the last dimension of the first with the second-to-last of the second.
        # So jnp.dot([B,S,E], [S,num_experts]) would give [B,num_experts] if E==S, but E=4096 and S=4096
        # Actually, looking at the shapes: x=[2,4096,4096], router_weights=[4096,8]
        # jnp.dot(x, router_weights) would give [2, 4096, 8] by contracting the last dim of x with the first dim of router_weights
        
        # So logits[b, s, e] = sum over E of x[b, s, E] * router_weights[E, e]
        # But wait, router_weights is [S, num_experts] = [4096, 8]
        # So logits[b, s, e] = sum over E of x[b, s, E] * router_weights[E, e]
        # But E=4096 and S=4096, so this works out dimensionally
        
        # Hmm, but that means we're using router_weights[E, e] for each E position, not router_weights[s, e]
        # Let me re-read the AST...
        
        # The AST says: logits = jnp.dot(x, router_weights)
        # x has shape [B, S, E] = [2, 4096, 4096]
        # router_weights has shape [S, num_experts] = [4096, 8]
        # jnp.dot contracts the last dimension of x (E=4096) with the first dimension of router_weights (S=4096)
        # Result is [B, num_experts] = [2, 8]? No wait...
        
        # Actually for 3D jnp.dot:
        # jnp.dot([a,b,c], [d,e,f]) where c==d gives [a,b,e,f]
        # So jnp.dot([2,4096,4096], [4096,8]) gives [2,4096,8]
        
        # So logits[b, s, e] = sum over E of x[b, s, E] * router_weights[E, e]
        # This is independent of s! That seems wrong...
        
        # Let me look at the shapes again. Maybe I'm misunderstanding.
        # x: [B, S, E] = [2, 4096, 4096]
        # router_weights: [S, num_experts] = [4096, 8]
        
        # For jnp.dot to work, we need the last dim of x to match the first dim of router_weights
        # E=4096, S=4096, so they match!
        # Result shape: [B, S, num_experts] = [2, 4096, 8]
        
        # So logits[b, s, e] = sum over E of x[b, s, E] * router_weights[E, e]
        # This is indeed independent of s in the router_weights indexing...
        
        # Wait, I think I need to be more careful. Let me trace through:
        # logits = jnp.dot(x, router_weights)
        # x[b, s, :] is a vector of length E=4096
        # router_weights[:, e] is a vector of length S=4096
        # So logits[b, s, e] = dot(x[b, s, :], router_weights[:, e])
        
        # Hmm, but that doesn't use router_weights[s, :] at all...
        
        # Let me just implement it as specified and see if it works.
        
        # Actually, I realize I should just implement the kernel following the AST semantics.
        # Let me think about this differently.
        
        # For the Pallas kernel, I need to process each (b, s) position.
        # At each position, I need to:
        # 1. Compute logits = dot(x[b, s, :], router_weights[:, e]) for each expert e
        #    But wait, the AST says logits = jnp.dot(x, router_weights)
        #    This gives [B, S, num_experts]
        
        # Let me just implement the full computation in the kernel.
        
        # Actually, looking at the shapes more carefully:
        # The AST shows that logits has shape [B, S, num_experts] after the dot product.
        # Then top_k is applied to get top_k_logits and top_k_indices.
        
        # For a Pallas kernel, I need to think about how to parallelize this.
        # The output is [B, S, E], so I can parallelize over B and S.
        
        # But the computation involves:
        # 1. Computing logits for all experts at position (b, s)
        # 2. Getting top-k indices
        # 3. Computing einsums with expert kernels
        
        # The einsum operations are:
        # gate_out = einsum("bse,nem->bsnm", x, expert_gate_kernels)
        # up_out = einsum("bse,nem->bsnm", x, expert_up_kernels)
        # hidden = gate_out * up_out
        # expert_outputs = einsum("bsnm,nme->bsne", hidden, expert_down_kernels)
        
        # These are all batched operations that can be computed for each (b, s) position.
        
        # Let me implement this step by step.
        
        # For simplicity, I'll use jnp operations inside the kernel.
        # The Pallas kernel will parallelize over batch and sequence dimensions.
        
        # Get the input slices for this position
        x_bs = x_ref[b, s, :]  # [E]
        
        # Compute logits for this position
        # logits[b, s, e] = sum over E of x[b, s, E] * router_weights[E, e]
        # But we need all logits for top-k, so compute for all experts
        logits = jnp.zeros(num_experts, dtype=jnp.float32)
        for e in range(num_experts):
            logits = logits.at[e].set(jnp.dot(x_bs, router_weights_ref[:, e]))
        
        # Get top-k indices and values
        top_k_logits, top_k_indices = jax.lax.top_k(logits, num_experts_per_tok)
        
        # Compute router probs
        router_probs = jax.nn.softmax(top_k_logits, axis=-1)
        
        # Compute gate_out = silu(einsum("bse,nem->bsnm", x, expert_gate_kernels))
        # For position (b, s), this is silu(einsum("e,nem->snm", x_bs, expert_gate_kernels))
        # einsum("e,nem->snm" means: for each expert n and m, sum over e of x[e] * gate_kernels[n, e, m]
        gate_out = jnp.zeros((num_experts, mlp_dim), dtype=jnp.float32)
        for n in range(num_experts):
            for m in range(mlp_dim):
                gate_out = gate_out.at[n, m].set(jnp.dot(x_bs, expert_gate_kernels_ref[n, :, m]))
        gate_out = jax.nn.silu(gate_out)
        
        # Compute up_out = einsum("bse,nem->bsnm", x, expert_up_kernels)
        up_out = jnp.zeros((num_experts, mlp_dim), dtype=jnp.float32)
        for n in range(num_experts):
            for m in range(mlp_dim):
                up_out = up_out.at[n, m].set(jnp.dot(x_bs, expert_up_kernels_ref[n, :, m]))
        
        # hidden = gate_out * up_out
        hidden = gate_out * up_out
        
        # expert_outputs = einsum("bsnm,nme->bsne", hidden, expert_down_kernels)
        # For position (b, s), this is einsum("snm,nme->sne", hidden, expert_down_kernels)
        # Result shape: [num_experts, E]
        expert_outputs = jnp.zeros((num_experts, E), dtype=jnp.float32)
        for n in range(num_experts):
            for e_out in range(E):
                expert_outputs = expert_outputs.at[n, e_out].set(jnp.dot(hidden[n, :], expert_down_kernels_ref[n, :, e_out]))
        
        # one_hot = one_hot(top_k_indices, N) where N = S - 1 = 4095
        N = S - 1
        one_hot = jax.nn.one_hot(top_k_indices, N)
        
        # weighted = one_hot * router_probs[..., None]
        weighted = one_hot * router_probs[:, None]
        
        # expert_weights = sum(weighted, axis=2)
        expert_weights = jnp.sum(weighted, axis=1)
        
        # output = einsum("bsne,bsn->bse", expert_outputs, expert_weights)
        # For position (b, s), this is einsum("sne,sn->se", expert_outputs, expert_weights)
        output = jnp.zeros(E, dtype=jnp.float32)
        for e_out in range(E):
            output = output.at[e_out].set(jnp.dot(expert_outputs[:, e_out], expert_weights))
        
        # Write output
        out_ref[b, s, :] = output
    
    # Define grid and specs
    grid = (B, S)
    
    def kernel(b, s):
        return (b, s, slice(None))
    
    return pl.pallas_call(
        sparse_moe_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.float32),
        grid=grid,
        in_specs=(
            pl.BlockSpec((1, 1, E), lambda b, s, _: (b, s, 0)),
            pl.BlockSpec((1, num_experts), lambda b, s, _: (0, 0)),
            pl.BlockSpec((num_experts, E, mlp_dim), lambda b, s, n, e, m: (n, e, m)),
            pl.BlockSpec((num_experts, E, mlp_dim), lambda b, s, n, e, m: (n, e, m)),
            pl.BlockSpec((num_experts, mlp_dim, E), lambda b, s, n, m, e: (n, m, e)),
        ),
        out_specs=pl.BlockSpec((1, 1, E), lambda b, s, _: (b, s, 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, router_weights, expert_gate_kernels, expert_up_kernels, expert_down_kernels)
