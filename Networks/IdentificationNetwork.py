import torch
import torch.nn as nn

class IdentificationNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_states):
        super(IdentificationNet, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_states = n_states
        self.last_state = None
        self.state = None  # Initialize this after seeing the first batch
        self.input_fc = nn.Linear(n_inputs, n_states)
        self.state_fc = nn.Linear(n_states+n_inputs, n_states)
        self.output_fc = nn.Linear(n_states, n_outputs)
        self.lstm = nn.LSTM(n_inputs, n_states, 1, batch_first=True)

    def forward(self, inputs):
        # Change of states
        #self.update_state(inputs.shape)
        batch_size = inputs.shape[0]
        if self.state is None:
           self.state = (torch.zeros(1, batch_size, self.n_states).to("cuda:0").float(),
                  torch.zeros(1, batch_size, self.n_states).to("cuda:0").float())
        x, h = self.lstm(inputs, self.state)
        self.state = [h[0].detach(), h[1].detach()]
        outs = self.output_fc(x)
        return outs

    def process_sequence(self,inputs):
        outs = []
        self.state = torch.zeros((inputs.shape[0], self.n_states)).float().to("cuda:0")
        for i in range(inputs.shape[1]):
            curr_input = inputs[:, i]
            self.state = self.state_fc(torch.cat((curr_input, torch.clone(self.state.detach())), dim=1))
            self.state = torch.tanh(self.state)
            out = self.output_fc(self.state)
            outs.append(out)
        # Update states here
        return torch.stack(outs).float()

    def update_state(self, shape):
        if self.last_state is not None:
            self.state = self.last_state
        else:
            self.state = torch.zeros((shape[0], self.n_states)).float().to("cuda:0")

    def reset(self):
        # if self.total_iterations > 500:
        #     self.output_feedback = True
        self.state = None