from torch import nn


class GenericNet(nn.Module):
    def __init__(self, n_in=10, n_out=1, n_hidden=1, hidden_neurons=80, nonlin=nn.ReLU6()):
        super().__init__()
        self.inp = nn.Linear(n_in, hidden_neurons)
        self.hid = [nn.Linear(hidden_neurons, hidden_neurons) for i in range(n_hidden)]
        self.out = nn.Linear(hidden_neurons, n_out)
        self.nl = nonlin

    def forward(self, xb):
        z = self.inp(xb)
        z = self.nl(z)
        for layer in self.hid:
            z = layer(z)
            z = self.nl(z)
        z = self.out(z)
        return z

class SISONet(GenericNet):

    def __init__(self, n_hidden=1, hidden_neurons=80, nonlin=nn.ReLU6()):
        GenericNet.__init__(self, n_in=1, n_out=1, n_hidden=n_hidden, hidden_neurons=hidden_neurons, nonlin=nonlin)



