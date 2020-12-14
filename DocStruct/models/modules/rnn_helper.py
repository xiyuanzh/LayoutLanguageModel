import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, input_channel, hidden_dim, bidirectional=True, batch_first=True):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(input_channel, hidden_dim, bidirectional=bidirectional, batch_first=batch_first)
        # init weight
        nn.init.xavier_normal_(self.rnn.weight_hh_l0)
        nn.init.xavier_normal_(self.rnn.weight_ih_l0)
        self.rnn.bias_hh_l0.data.zero_()
        self.rnn.bias_ih_l0.data.zero_()
        if bidirectional:
            nn.init.xavier_normal_(self.rnn.weight_hh_l0_reverse)
            nn.init.xavier_normal_(self.rnn.weight_ih_l0_reverse)
            self.rnn.bias_hh_l0_reverse.data.zero_()
            self.rnn.bias_ih_l0_reverse.data.zero_()

    def forward(self, input):
        output, _ = self.rnn(input)
        return output  # output: T, b, 2*hidden_dim
