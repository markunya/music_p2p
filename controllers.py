import abc
import torch

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def default_calc_hidden_states(self, query, key, value):
        vk = torch.matmul(value, key)
        hidden_states = torch.matmul(vk, query)
        return hidden_states
    
    @abc.abstractmethod
    def forward(self, query, key, value):
        raise NotImplementedError

    def __call__(self, query, key, value):
        h = query.shape[0]
        end = h // 2 # because of guidence, second part with uncond prompts
        hidden_states = torch.cat([
            self.forward(query[:end], key[:end], value[:end]),
            self.default_calc_hidden_states(query[end:], key[end:], value[end:])
        ])
        
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

        return hidden_states
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
