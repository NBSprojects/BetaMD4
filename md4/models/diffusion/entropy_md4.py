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

"""Entropy-driven Masked Diffusion Model (EntropyMD4)."""

from typing import Any, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from md4 import utils
from md4.models import backward
from md4.models import utils as model_utils

import ml_collections

tfd = tfp.distributions
tfm = tfp.math


class EntropyMD4(nn.Module):
  """
  Masked diffusion model with a dynamically computed, entropy-driven
  noise schedule for each token.
  """

  # --- Configuration Attributes (from config file) ---
  config: ml_collections.ConfigDict
  data_shape: tuple[int, ...]
  cont_time: bool
  timesteps: int
  feature_dim: int
  num_heads: int
  n_layers: int
  n_dit_layers: int
  dit_num_heads: int
  dit_hidden_size: int
  ch_mult: Sequence[int]
  vocab_size: int
  dropout_rate: float
  use_attn_dropout: bool
  mlp_type: str
  depth_scaled_init: bool
  cond_type: str
  outside_embed: bool
  time_features: str
  classes: int
  # --- New attributes specific to our model ---
  entropy_k: int
  antithetic_time_sampling: bool = True
  t1: float = 1e-3  # Minimum time

  def setup(self):
    """Initializes the trainable parts of the model."""
    if self.classes > 0:
      self.cond_embeddings = nn.Embed(self.classes, self.feature_dim)

    self.classifier = backward.DiscreteClassifier(
        n_layers=self.n_layers,
        n_dit_layers=self.n_dit_layers,
        dit_num_heads=self.dit_num_heads,
        dit_hidden_size=self.dit_hidden_size,
        ch_mult=self.ch_mult,
        feature_dim=self.feature_dim,
        num_heads=self.num_heads,
        vocab_size=self.vocab_size,
        dropout_rate=self.dropout_rate,
        use_attn_dropout=self.use_attn_dropout,
        mlp_type=self.mlp_type,
        depth_scaled_init=self.depth_scaled_init,
        cond_type=self.cond_type,
        outside_embed=self.outside_embed,
    )

    # Load auxiliary models M1 and M2 using their dedicated loaders
    print('Loading m1 model...')
    self.m1_model, self.m1_variables = model_utils.load_m1_model(
        config=self.config.m1_config,
        weights_path=self.config.m1_weights_path,
        input_shape=self.data_shape,
    )
    print('Successfully loaded m1 model')

    print('Loading m2 model...')
    self.m2_model, self.m2_variables = model_utils.load_m2_model(
        config=self.config.m2_config,
        weights_path=self.config.m2_weights_path,
        input_shape=self.data_shape,
    )
    print('Successfully loaded m2 model')

  def get_cond_embedding(self, conditioning):
    if conditioning is not None and self.classes > 0:
      return self.cond_embeddings(conditioning)
    return None

  def predict_x(self, zt, t, cond=None, train=False):
    """Wrapper to call the classifier."""
    t_feat = t if self.time_features != "none" else None
    return self.classifier(zt, t=t_feat, cond=cond, train=train)

  def recon_loss(self):
    """Dummy reconstruction loss for API compatibility."""
    return jnp.array(0.0)

  def latent_loss(self):
    """Dummy latent loss for API compatibility."""
    return jnp.array(0.0)

  def _calculate_beta_params(
      self,
      x: jnp.ndarray,
      train: bool = False,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Calculates Beta distribution parameters (a, b) from token entropy."""
    b, l = x.shape
    m = l // self.entropy_k

    x_repeated = jnp.repeat(x, m, axis=0)

    indices = jnp.arange(l)
    i_mod_m = jnp.arange(m)
    mask_template = (indices % m)[:, None] == i_mod_m
    full_mask = jnp.tile(mask_template.T, (b, 1))

    masked_batch = jnp.where(full_mask, self.vocab_size, x_repeated)

    t_for_m1 = jnp.array(self.t1)

    logits, _ = self.m1_model.apply(
        self.m1_variables,
        masked_batch,
        t=t_for_m1,
        cond=None,
        train=False,
        method=self.m1_model.predict_x,
    )

    n_indep_axes = logits.ndim - 2  # Pour le texte, shape est (B, L, V), donc ndim=3 -> n_indep_axes=1
    dist_m1 = tfd.Independent(tfd.Categorical(logits=logits), n_indep_axes)

    token_entropies = dist_m1.entropy()
    entropy_values = jnp.where(full_mask, token_entropies, 0.0)

    entropy_sequences = jnp.sum(entropy_values.reshape(b, m, l), axis=1)

    sum_entropy = jnp.sum(entropy_sequences, axis=-1, keepdims=True)
    normalized_entropies = entropy_sequences / (sum_entropy + 1e-6)

    a, b = self.m2_model.apply(self.m2_variables, normalized_entropies, train=train)

    return a, b

  def diffusion_loss(self, t, x, zt, cond=None, train=False):
    """Calculates the diffusion loss based on the masked input."""
    logits, _ = self.predict_x(zt, t, cond=cond, train=train)
    log_p = jax.nn.log_softmax(logits, axis=-1)
    one_hot_x = jax.nn.one_hot(x, self.vocab_size)
    neg_cross_ent = -jnp.sum(one_hot_x * log_p, axis=-1)
    mask = (zt == self.vocab_size).astype(jnp.float32)
    loss = jnp.sum(neg_cross_ent * mask, axis=list(range(x.ndim)[1:]))
    return loss

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      cond: jnp.ndarray | None = None,
      train: bool = False,
  ):
    """The main forward pass of the model."""
    if self.m1_model is None or self.m1_variables is None or self.m2_model is None or self.m2_variables is None:
      dummy_zt = jnp.zeros_like(x)
      dummy_t = jnp.ones(x.shape[0])
      self.predict_x(
          dummy_zt,
          t=dummy_t,
          cond=self.get_cond_embedding(cond),
          train=train,
      )
      return {
          "loss": jnp.array(0.0),
          "loss_diff": jnp.array(0.0),
          "loss_prior": jnp.array(0.0),
          "loss_recon": jnp.array(0.0),
      }

    rng = self.make_rng("sample")
    bs = x.shape[0]
    cond_embedding = self.get_cond_embedding(cond)

    a, b = jax.lax.stop_gradient(
        self._calculate_beta_params(x, train=train)
    )

    if self.antithetic_time_sampling:
      t0 = jax.random.uniform(rng)
      t = jnp.mod(t0 + jnp.arange(0.0, 1.0, step=1.0 / bs), 1.0)
    else:
      t = jax.random.uniform(rng, shape=[bs])
    t = (1.0 - self.t1) * t + self.t1

    un_mask_prob = tfm.betainc(utils.reverse_broadcast(t, a.ndim), a, b)
    un_mask = jax.random.bernoulli(rng, un_mask_prob, x.shape)
    zt = jnp.where(un_mask, x, self.vocab_size)

    loss_diff = self.diffusion_loss(t, x, zt, cond=cond_embedding, train=train)
    loss_prior = self.latent_loss()
    loss_recon = self.recon_loss()

    loss = loss_diff.mean() + loss_prior + loss_recon
    loss_diff_mean = loss_diff.mean()

    model_stats = {
        "loss": loss,
        "loss_nelbo": loss,
        "loss_diff": loss_diff_mean,
        "loss_prior": loss_prior,
        "loss_recon": loss_recon,
    }
    model_stats = utils.loss2bpt(model_stats, self.data_shape)
    return model_stats 