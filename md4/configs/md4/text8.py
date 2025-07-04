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

r"""A config for training MD4 on text8."""

from collections import abc

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
  """Default config."""

  config = config_dict.ConfigDict()

  # dataset configs
  config.vocab_size = 27
  config.dataset = "text8"
  config.classes = -1

  config.task_type = "text"  # text or image
  config.model_type = "md4"
  config.data_shape = (256,)

  # timesteps: int or None
  config.timesteps = 1000
  # linear, cosine, poly[exponent], e.g., poly3
  config.noise_schedule = "linear"
  config.outside_embed = True
  # t or none (removes time dependence)
  config.time_features = "t"
  config.cont_time = True

  config.feature_dim = 64
  config.n_layers = 12
  config.ch_mult = (1,)  # not used
  config.n_dit_layers = 0  # not used
  config.dit_num_heads = 12  # not used
  config.dit_hidden_size = 768  # not used
  config.dropout_rate = 0.0

  config.num_heads = 12
  config.mlp_type = "glu"
  config.depth_scaled_init = True
  config.cond_type = "adaln_zero"

  config.learning_rate = 3e-4
  config.learning_rate_schedule = "cosine"
  config.warmup_steps = 2000
  config.weight_decay = 0.0
  config.clip = 0.0
  config.b2 = 0.999
  config.num_epochs = -1
  config.ema_rate = 0.9999
  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs.
  config.num_train_steps = 1_000_000
  # Evaluates for a full epoch if num_eval_steps==-1.
  config.num_eval_steps = -1
  config.batch_size = 512
  config.num_microbatches = 1
  config.per_device_batch_size = -1
  # If batches should be added to evaluate the entire dataset.
  config.eval_pad_last_batch = False
  config.check_nans = False

  # Sampling
  # ancestral, mean, or topp
  config.sampler = "ancestral"
  # uniform, cosine
  config.sampling_grid = "cosine"
  # for topp sampler
  config.topp = 0.98

  config.log_loss_every_steps = 500
  config.eval_every_steps = 5000
  config.checkpoint_every_steps = 5000
  config.checkpoint_keep_period = 10000

  # Single integer or tuple. If None will use (XManager ID, work unit).
  config.seed = 42

  # Number of workers for Grain loaders.
  config.grain_num_workers = 15

  config.trial = 0  # Dummy for repeated runs.
  config.test_in_colab = False

  # training
  config.train.optimizer = 'adam'
  config.train.learning_rate = 2e-4
  config.train.warmup_steps = 5000
  config.train.weight_decay = 0.0
  config.train.ema_decay = 0.9999
  config.train.batch_size = 128
  config.train.num_steps = 700001

  # model
  config.model = ml_collections.config_dict.ConfigDict()
  config.model.model_name = 'entropy_md4'
  config.model.noise_schedule = 'linear'
  config.model.timesteps = 1000
  config.model.cont_time = True
  config.model.depth_scaled_init = False
  config.model.cond_type = 'adaln_zero'
  config.model.classes = 0

  # --- New parameters for EntropyMD4 ---
  config.model.entropy_k = 4
  config.model.m1_weights_path = 'trained_models/md4_text8_step_70000.msgpack'
  config.model.m2_weights_path = 'trained_models/jax_model_weights.msgpack'

  # --- M2 Model (JaxBetaParamMLP) Configuration ---
  config.model.m2_config = ml_collections.config_dict.ConfigDict()
  config.model.m2_config.model_name = 'beta_mlp'
  config.model.m2_config.hidden_layers = [500, 500, 500]
  config.model.m2_config.output_size = config.data_shape[0] * 2

  return config


# By default, the launcher calls `