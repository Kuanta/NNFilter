import torch
import torch.nn as nn

class ARNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, out_index):
        super(ARNet, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.past_preds = None
        self.out_index = out_index
        self.fc1 = nn.Linear(7, 1, bias=False)  # Single layer
        #self.fc1.weight = nn.Parameter(torch.tensor([[0, 0.1113, -0.1775, 0.0671, 2.7563, -2.5360, 0.7788]]))
        #self.fc1.weight.requires_grad = False
        # At initial iterations, it may be better to turn off output feedback where prediction at k used at prediction at k+1
        self.output_feedback = False
        self.total_iterations = 0
        self.last_out = None

    def forward(self, inputs):
        if not self.output_feedback or self.past_preds is None:
            self.past_preds = inputs[:, :, self.out_index:]
        else:
            self.update_past_preds(self.last_out)
        x = torch.cat((inputs[:, :, :self.out_index], self.past_preds), dim=2)
        out = self.fc1(x)
        self.total_iterations += 1
        self.last_out = torch.clone(out)
        return out

    def update_past_preds(self, out):
        if self.output_feedback is True:
            self.past_preds[:, :, 1:] = torch.clone(self.past_preds[:, :, :-1])  # Shift the columns to the right by one
            self.past_preds[:, :, 0] = out.detach()[:, :, 0]  # TODO: For now, output must be a one dimensional value

    def reset(self):
        # if self.total_iterations > 500:
        #     self.output_feedback = True
        self.past_preds = None