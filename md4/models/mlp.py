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

"""A simple MLP module for use as an auxiliary model."""

from typing import Sequence
import flax.linen as nn

class MLP(nn.Module):
  """A simple Multi-Layer Perceptron (MLP) with BatchNorm and ReLU."""
  features: Sequence[int]

  @nn.compact
  def __call__(self, x, train: bool = False):
    """
    Args:
      x: The input tensor.
      train: Whether the model is in training mode. This is crucial for
        BatchNorm. Since this MLP is used as a fixed model during the main
        training, we will always call it with `train=False`.
    """
    use_running_average = not train

    # Iterate through all but the last layer size to build the hidden layers
    for i, feat in enumerate(self.features[:-1]):
      x = nn.Dense(feat, name=f'dense_{i}')(x)
      # BatchNorm in Flax requires explicit state handling.
      # use_running_average=True ensures we use the learned stats and don't
      # try to update them, which is exactly what we want for a fixed model.
      x = nn.BatchNorm(
          use_running_average=use_running_average, name=f'batchnorm_{i}'
      )(x)
      x = nn.relu(x)

    # The final output layer, without activation here
    x = nn.Dense(self.features[-1], name='dense_output')(x)

    # Apply Softplus to ensure outputs are > 0, and add a small epsilon
    # for numerical stability, matching the original PyTorch model.
    x = nn.softplus(x) + 1e-6
    
    # Reshape and split the output into 'a' and 'b' parameters.
    # The output of the last Dense layer is (batch, seq_len * 2).
    # We reshape it to (batch, seq_len, 2) to easily split.
    output_shape = x.shape[:-1] + (-1, 2)
    x = x.reshape(output_shape)
    
    # We don't split into two variables here, we return the combined tensor.
    # The caller (`EntropyMD4`) will be responsible for slicing a and b.
    # This keeps the MLP model more generic.
    return x 