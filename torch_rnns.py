import torch

class ElmanRNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, device):
        super().__init__()

        self.device = device

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = torch.nn.RNN(input_size, hidden_size, bias=False)  # , dropout=0.5)#, dropout=0.5)#, dropout=0.5)
        self.out_linear = torch.nn.Linear(self.hidden_size, output_size, bias=False)

    def forward(self, input, hidden, verbose=False):
        rnn_out, hidden1 = self.rnn(input, hidden)
        output = self.out_linear(rnn_out)
        if not verbose:
            return output, hidden1
        else:
            return rnn_out, output, output, hidden1

    def init_hidden(self, batch_size):
        hidden1_state = torch.randn(1, batch_size, self.hidden_size, device=self.device)
        # cell1_state = torch.randn(1, batch_size, self.hidden_size, device=device)

        # hc1 = (hidden1_state, cell1_state)
        return hidden1_state
        # return (hidden1_state, cell1_state)  # hidden1_state


class LstmRNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, device):
        super().__init__()

        self.device = device

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = torch.nn.LSTM(input_size, hidden_size, bias=False)  # , dropout=0.5)#, dropout=0.5)#, dropout=0.5)
        self.out_linear = torch.nn.Linear(self.hidden_size, output_size, bias=False)

    def forward(self, input, hidden, verbose=False):
        rnn_out, hidden1 = self.rnn(input, hidden)
        output = self.out_linear(rnn_out)
        if not verbose:
            return output, hidden1
        else:
            return rnn_out, output, output, hidden1

    def init_hidden(self, batch_size):
        hidden1_state = torch.randn(1, batch_size, self.hidden_size, device=self.device)
        cell1_state = torch.randn(1, batch_size, self.hidden_size, device=self.device)

        # hc1 = (hidden1_state, cell1_state)
        # return hidden1_state
        return (hidden1_state, cell1_state)  # hidden1_state
