import torch
import torch.nn as nn


class Amalgamator(nn.Module):
    def __init__(self, num_teachers=15, student_hidden_size=1024):
        super(Amalgamator, self).__init__()
        self.fc1 = nn.Linear(student_hidden_size, num_teachers)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(num_teachers*2, num_teachers)
        
        
    def forward(self, s_input, t_uncertainty):
        s_outputs = self.fc1(s_input)
        s_t = self.activation(torch.cat((s_outputs, t_uncertainty), dim=-1))
        t_weights = self.fc2(s_t)

        return t_weights
