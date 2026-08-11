import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias):
    def kernel(x_ref, w_ref, b_ref, out_ref):
        x = x_ref[...]
        w = w_ref[...]
        b = b_ref[...]
        # Transpose x: (0,2,3,4,1) -> wait original: x transpose to (0,2,3,4,1)?
        # From AST: transpose x with (0,2,3,4,1)
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        # Transpose weight: (2,3,4,1,0)
        kernel_w = jnp.transpose(w, (2, 3, 4, 1, 0))
        # Conv
        x = lax.conv_general_dilated(
            x, kernel_w,
            window_strides=(1, 1, 1),
            padding=((0, 0), (0, 0), (0, 0)),
            dimension_numbers=("NDHWC", "DHWIO", "NDHWC")
        )
        # Add bias reshaped
        b_reshaped = jnp.reshape(b, (1, 1, 1, 1, -1))
        # Wait AST: reshape bias with (1,1,1,1,-1) then add
        # Actually AST: reshape(bias, (1,1,1,1,-1))? Let's check: args are 1,1,1,1, UnaryOp(USub(1)) -> (1,1,1,1,-1)
        b_reshaped = jnp.reshape(b, (1, 1, 1, 1, -1))
        x = x + b_reshaped
        # Mish: x * tanh(log(1 + exp(x)))
        # Actually AST: tanh(log(1 + exp(x))) * x? Let's parse:
        # Mult(left=x, right=tanh(log(1 + exp(x))))
        # So x * tanh(log(1 + exp(x)))
        mish = jnp.tanh(jnp.log(1 + jnp.exp(x)))
        x = x * mish
        # Tanh
        x = jnp.tanh(x)
        # Transpose back: (0,4,1,2,3)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        out_ref[...] = x

    out_shape = jax.ShapeDtypeStruct((16, 64, 30, 62, 62), jnp.bfloat16)
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(),
        in_specs=(pl.no_block_spec, pl.no_block_spec, pl.no_block_spec),
        out_specs=pl.no_block_spec,
    )(x, weight, bias)
