import flax.linen as nn
import jax.numpy as jnp


class MLP(nn.Module):
    x = nn.Dense(features=self.output_dim)(x)
    return x


class JaxBetaParamMLP(nn.Module):
  """MLP with BatchNorm to predict Beta distribution parameters."""

  output_size: int
  hidden_layers: list

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = True) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Forward pass with BatchNorm handling."""
    for i, hidden_size in enumerate(self.hidden_layers):
      x = nn.Dense(features=hidden_size, name=f"dense_{i}")(x)
      x = nn.BatchNorm(use_running_average=not train, name=f"bn_{i}")(x)
      x = nn.relu(x)

    x = nn.Dense(features=self.output_size, name="dense_out")(x)
    params = nn.softplus(x) + 1e-6

    output_dim = self.output_size // 2
    a_params, b_params = jnp.split(params, [output_dim], axis=-1)
    return a_params, b_params 
