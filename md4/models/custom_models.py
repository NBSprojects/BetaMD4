# md4/models/custom_models.py

import flax.linen as nn
from ml_collections import config_dict
import jax
import jax.numpy as jnp
import math
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

from md4.models.diffusion import md4 as md4_lib
from md4.models import backward as backward_lib

class M1_MDM_for_Inference(nn.Module):
    """
    Cette classe reproduit la structure du modèle MD4 et inclut maintenant
    la logique d'échantillonnage correcte avec la grille cosinus.
    """
    config: config_dict.ConfigDict

    def setup(self):
        """Initialise les mêmes sous-modules que le modèle original."""
        self.noise_schedule = md4_lib.MaskingSchedule(
            data_shape=self.config.data_shape,
            schedule_fn_type=self.config.noise_schedule
        )
        self.classifier = backward_lib.DiscreteClassifier(
            n_layers=self.config.n_layers,
            feature_dim=self.config.feature_dim,
            num_heads=self.config.num_heads,
            vocab_size=self.config.vocab_size,
            dropout_rate=0.0,
            mlp_type=self.config.mlp_type,
            cond_type=self.config.cond_type,
            outside_embed=self.config.outside_embed,
        )

    def __call__(self, zt, t):
        """Appelle le classifieur pour l'inférence."""
        logits, _ = self.classifier(zt, t=t, cond=None, train=False)
        return logits

    def get_sampling_grid(self, i, timesteps):
        """
        Calcule les pas de temps s et t en utilisant la grille spécifiée dans la config.
        Ceci est la fonction manquante qui causait le problème.
        """
        t = (timesteps - i) / timesteps
        s = t - 1 / timesteps
        if self.config.sampling_grid == 'cosine':
            t = jnp.cos(math.pi / 2.0 * (1.0 - t))
            s = jnp.cos(math.pi / 2.0 * (1.0 - s))
        return s, t

    def ancestral_sampling_loop(self, rng, batch_size, seq_len, timesteps, vocab_size):
        """
        Exécute la boucle de 'denoising' itérative.
        """
        zt = jnp.full((batch_size, seq_len), vocab_size, dtype=jnp.int32)
        
        def body_fn(i, zt):
            rng_body = jax.random.fold_in(rng, i)
            
            # Utilise maintenant la méthode de grille correcte.
            s_val, t_val = self.get_sampling_grid(i, timesteps)
            
            t_vec = jnp.ones((batch_size,)) * t_val
            logits = self(zt, t_vec)
            
            alpha_t = self.noise_schedule.alpha(t_val)
            alpha_s = self.noise_schedule.alpha(s_val)
            
            unmask_prob = (alpha_s - alpha_t) / (1 - alpha_t)
            
            probs_vocab = unmask_prob * nn.softmax(logits, axis=-1)
            probs_mask = jnp.ones(list(zt.shape) + [1]) * (1 - unmask_prob)
            probs = jnp.concatenate([probs_vocab, probs_mask], axis=-1)
            
            to_unmask = tfd.Categorical(probs=probs).sample(seed=rng_body)
            
            is_mask = (zt == vocab_size)
            zs = jnp.where(is_mask, to_unmask, zt)
            return zs

        final_state = jax.lax.fori_loop(0, timesteps, body_fn, zt)
        return final_state

class M2_MLP(nn.Module):
    """Le modèle M2 (MLP) reste inchangé."""
    hidden_dim: int = 256
    output_dim: int = 2

    @nn.compact
    def __call__(self, x, train=False):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        params = nn.Dense(features=self.output_dim)(x)
        return nn.softplus(params)