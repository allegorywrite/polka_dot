
import torch.nn as nn
from models.deepset import Deepset
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class PolkaDot(TorchModelV2, nn.Module):
  def __init__(self, obs_space, action_space, num_outputs, model_config, name):
    TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
    nn.Module.__init__(self)
    self.state_dim = 6
    self.deepset_latent_dim = 16

    self.action_model = FullyConnectedNetwork(
      Box(low=-1, high=1, shape=(OWN_OBS_VEC_SIZE, )), 
      action_space,
      num_outputs,
      model_config,
      name + "_action"
      )
    self.value_model = FullyConnectedNetwork(
      obs_space, 
      action_space,
      1, 
      model_config, 
      name + "_vf"
      )
    self.model_neighbors = DeepSet(self.state_dim, self.deepset_latent_dim)
    self._model_in = None
  
  def forward(self, input_dict, state, seq_lens):
    self._model_in = [input_dict["obs_flat"], state, seq_lens]
    output = self.action_model({"obs" : input_dict["obs"]["own_obs"]}, state, seq_lens)
    return output

  def value_function(self):
    value_out, _ = self.value_model({"obs": self._model_in[0]}, self._model_in[1], self._model_in[2])
    return torch.reshape(value_out, [-1])

  
