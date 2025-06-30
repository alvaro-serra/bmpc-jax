from functools import partial
from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp
import jax


class NormedLinear(nn.Module):
  embed_dim: int
  activation: Callable[[jax.Array], jax.Array] = None
  dropout_rate: Optional[float] = None

  kernel_init: Callable = nn.initializers.truncated_normal(0.02)
  dtype: jnp.dtype = jnp.float32  # Switch this to bfloat16 for speed

  @nn.compact
  def __call__(self,
               x: jax.Array,
               train: bool = True) -> jax.Array:
    x = nn.Dense(
        features=self.embed_dim,
        kernel_init=self.kernel_init,
        dtype=self.dtype,
    )(x)

    x = nn.LayerNorm()(x)
    if self.activation is not None:
      x = self.activation(x)

    if self.dropout_rate is not None and self.dropout_rate > 0:
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    return x
