import torch

class MdnRnn(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_gaussians=5):
        super(MdnRnn, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_gaussians = num_gaussians
        self.rnn = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_gaussians + 2 * self.output_size * num_gaussians)

    def forward(self, input, hidden):
        rnn_out, hidden = self.rnn(input, hidden)
        batch_size = rnn_out.size(0)
        flat_out = self.fc(rnn_out)
        pi = flat_out[:, :, :self.num_gaussians]
        sigma = flat_out[:, :, self.num_gaussians:self.num_gaussians + self.output_size * self.num_gaussians]
        sigma = sigma.view(batch_size, -1, self.num_gaussians, self.output_size)
        sigma = torch.exp(sigma)
        mu = flat_out[:, :, self.num_gaussians + self.output_size * self.num_gaussians:]
        mu = mu.view(batch_size, -1, self.num_gaussians, self.output_size)
        pi = torch.softmax(pi, dim=-1)

        return pi, sigma, mu, hidden