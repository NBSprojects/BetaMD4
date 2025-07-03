# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Model utils."""

import ml_collections
import jax
import jax.numpy as jnp
import logging
from pathlib import Path as epath
from flax import serialization
import flax.linen as nn
import flax.core

from md4.models.diffusion import genmd4
from md4.models.diffusion import md4
from md4.models.diffusion import entropy_md4
from md4.models.networks import mlp as mlp_network


def get_model(config: ml_collections.ConfigDict):
  """Get model instances."""
  if config.model_type == "md4":
    return md4.MD4(
        config.data_shape,
        cont_time=config.cont_time,
        timesteps=config.timesteps,
        feature_dim=config.feature_dim,
        num_heads=config.num_heads,
        n_layers=config.n_layers,
        n_dit_layers=config.n_dit_layers,
        dit_num_heads=config.dit_num_heads,
        dit_hidden_size=config.dit_hidden_size,
        ch_mult=config.ch_mult,
        vocab_size=config.vocab_size,
        noise_schedule_type=config.noise_schedule,
        dropout_rate=config.dropout_rate,
        use_attn_dropout=config.get("use_attn_dropout", True),
        mlp_type=config.mlp_type,
        depth_scaled_init=config.depth_scaled_init,
        cond_type=config.cond_type,
        outside_embed=config.outside_embed,
        time_features=config.time_features,
        classes=config.classes,
        sampler=config.sampler,
        sampling_grid=config.sampling_grid,
        topp=config.topp,
        model_sharding=config.get("model_sharding", False),
    )
  elif config.model_type == "genmd4":
    return genmd4.GenMD4(
        config.data_shape,
        cont_time=config.cont_time,
        timesteps=config.timesteps,
        feature_dim=config.feature_dim,
        num_heads=config.num_heads,
        n_layers=config.n_layers,
        n_dit_layers=config.n_dit_layers,
        dit_num_heads=config.dit_num_heads,
        dit_hidden_size=config.dit_hidden_size,
        ch_mult=config.ch_mult,
        vocab_size=config.vocab_size,
        noise_schedule_type=config.noise_schedule,
        power_init=config.power_init,
        dropout_rate=config.dropout_rate,
        use_attn_dropout=config.get("use_attn_dropout", True),
        mlp_type=config.mlp_type,
        depth_scaled_init=config.depth_scaled_init,
        cond_type=config.cond_type,
        outside_embed=config.outside_embed,
        time_features=config.time_features,
    )
  elif config.model_type == "entropy_md4":
    return entropy_md4.EntropyMD4(
        data_shape=config.data_shape,
        cont_time=config.cont_time,
        timesteps=config.timesteps,
        feature_dim=config.feature_dim,
        num_heads=config.num_heads,
        n_layers=config.n_layers,
        n_dit_layers=config.n_dit_layers,
        dit_num_heads=config.dit_num_heads,
        dit_hidden_size=config.dit_hidden_size,
        ch_mult=config.ch_mult,
        vocab_size=config.vocab_size,
        dropout_rate=config.dropout_rate,
        use_attn_dropout=config.get("use_attn_dropout", True),
        mlp_type=config.mlp_type,
        depth_scaled_init=config.depth_scaled_init,
        cond_type=config.cond_type,
        outside_embed=config.outside_embed,
        time_features=config.time_features,
        classes=config.classes,
        entropy_k=config.entropy_k,
        t1=config.t1,
        antithetic_time_sampling=config.antithetic_time_sampling,
        config=config,
    )
  elif config.model_type == "beta_mlp":
    return mlp_network.JaxBetaParamMLP(
        output_size=config.output_size, hidden_layers=config.features
    )
  else:
    raise NotImplementedError(f"No model found for {config.model_type}")


def load_m1_model(
    config: ml_collections.ConfigDict,
    weights_path: str,
    input_shape: tuple[int, ...],
) -> tuple[nn.Module, dict]:
  """Loads the M1 model (MD4) which has raw params in its weight file."""
  model = get_model(config)
  rng = jax.random.PRNGKey(0)
  dummy_input = jnp.ones((1,) + input_shape, dtype="int32")

  # Initialize to get the variable structure.
  variables = model.init(rng, dummy_input, train=False)
  
  params_struct = variables.pop('params')
  # The rest is model state (e.g., batch_stats), though M1 might not have it.
  model_state_struct = variables

  if not epath(weights_path).exists():
    raise FileNotFoundError(f"Model weights not found at: {weights_path}")

  with epath(weights_path).open("rb") as f:
    # Load raw bytes into the 'params' structure.
    loaded_params = serialization.from_bytes(params_struct, f.read())

  # Re-assemble the full dictionary.
  final_variables = {'params': loaded_params, **model_state_struct}
  
  return model, flax.core.freeze(final_variables)


def load_m2_model(
    config: ml_collections.ConfigDict,
    weights_path: str,
    input_shape: tuple[int, ...],
    is_text_model: bool = True,
    init_kwargs: dict | None = None,
) -> tuple[nn.Module, dict]:
  """Loads the M2 model (MLP) which has a full var dict in its weight file."""
  if init_kwargs is None:
    init_kwargs = {}
  logging.info("Loading model from config: %s", config)
  logging.info("Loading weights from: %s", weights_path)

  model = get_model(config)
  rng = jax.random.PRNGKey(0)

  if is_text_model:
    dummy_input = jnp.ones((1,) + input_shape, dtype="int32")
  else:  # For models taking float inputs like the MLP
    dummy_input = jnp.ones((1,) + input_shape, dtype=jnp.float32)

  # Initialize to get variable structure, M2 returns the dict directly.
  variables = model.init(rng, dummy_input, train=False, **init_kwargs)

  if not epath(weights_path).exists():
    raise FileNotFoundError(f"Model weights not found at: {weights_path}")

  with epath(weights_path).open("rb") as f:
    # The file contains the full variable dictionary.
    loaded_variables = serialization.from_bytes(variables, f.read())

  return model, loaded_variables
