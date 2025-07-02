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

r"""A config for training EntropyMD4 on text8."""

from collections import abc

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
  """Default config for EntropyMD4."""

  config = config_dict.ConfigDict()

  # --- Main model and dataset configs ---
  config.vocab_size = 27
  config.dataset = "text8"
  config.classes = -1

  config.task_type = "text"
  config.model_type = "entropy_md4"
  config.data_shape = (256,)

  # --- EntropyMD4 Specific Configs ---
  config.entropy_k = 16  # Sub-sampling factor for entropy calculation

  # Diffusion model parameters
  config.timesteps = 1000
  config.cont_time = True
  config.t1 = 1e-3
  config.antithetic_time_sampling = True
  
  # Main model architecture (the denoiser)
  config.feature_dim = 64
  config.n_layers = 12
  config.num_heads = 12
  config.mlp_type = "glu"
  config.depth_scaled_init = True
  config.cond_type = "adaln_zero"
  config.outside_embed = True
  config.time_features = "t"
  config.dropout_rate = 0.0
  config.use_attn_dropout = True
  config.ch_mult = (1,)
  config.n_dit_layers = 0
  config.dit_num_heads = 12
  config.dit_hidden_size = 768
  
  # --- Auxiliary Model M1 Config (Fixed Pre-trained Diffusion Model) ---
  # --- UPDATED based on inference.py ---
  config.m1_config = config_dict.ConfigDict()
  config.m1_config.model_type = "md4"
  config.m1_config.data_shape = config.data_shape
  config.m1_config.vocab_size = config.vocab_size
  config.m1_config.classes = config.classes
  config.m1_config.feature_dim = 64
  config.m1_config.n_layers = 8
  config.m1_config.num_heads = 6
  config.m1_config.outside_embed = True
  config.m1_config.noise_schedule = 'linear'
  config.m1_config.time_features = 't'
  config.m1_config.cont_time = True
  config.m1_config.cond_type = 'adaln_zero'
  config.m1_config.mlp_type = 'glu'
  config.m1_config.sampler = 'ancestral'
  config.m1_config.sampling_grid = 'cosine'
  config.m1_config.timesteps = 1000
  # --- Path to the pre-trained weights for M1 ---
  config.m1_weights_path = "trained_models/md4_text8_step_70000.msgpack"

  # --- Auxiliary Model M2 Config (Fixed Pre-trained MLP) ---
  config.m2_config = config_dict.ConfigDict()
  # NOTE: We will need to add 'mlp' to model_utils.get_model
  config.m2_config.model_type = "beta_mlp" 
  # Example MLP architecture
  config.m2_config.features = [500, 500, 500] 
  config.model.m2_config.output_size = config.data.data_shape[0] * 2
  # --- Path to the pre-trained weights for M2 ---
  config.m2_weights_path = "trained_models/dummy_entropy_mlp.msgpack"


  # --- Training Configs ---
  config.learning_rate = 3e-4
  config.learning_rate_schedule = "cosine"
  config.warmup_steps = 2000
  config.weight_decay = 0.0
  config.clip = 0.0
  config.b2 = 0.999
  config.num_epochs = -1
  config.ema_rate = 0.9999
  config.num_train_steps = 1_000_000
  config.num_eval_steps = -1
  config.batch_size = 64
  config.num_microbatches = 1
  config.per_device_batch_size = -1
  config.eval_pad_last_batch = False
  config.check_nans = False

  config.log_loss_every_steps = 500
  config.eval_every_steps = 5000
  config.checkpoint_every_steps = 5000
  config.checkpoint_keep_period = 10000

  config.seed = 42
  config.grain_num_workers = 15
  config.trial = 0

  return config
