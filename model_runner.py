from nn_model import ModelStateLogDTO
import torch

class ModelRunner:
    def __init__(self, model, char2ix, device):
        self.model = model
        self.char2ix = char2ix
        self.ix2char = {v: k for k, v in char2ix.items()}
        self.device = device

    def run_model_on_word(self, word):
        self.model.eval()
        states_log = []
        with torch.no_grad():
            word_ixed = [self.char2ix[c] for c in word if c in self.char2ix]
            sequence_length = len(word_ixed)
            batch_size = 1
            word_ixed_tensor = torch.Tensor(word_ixed).view(sequence_length, batch_size).long().to(self.device)

            src_ohe = torch.nn.functional.one_hot(word_ixed_tensor,
                                                  len(self.char2ix)).to(torch.float).to(self.device)

            hidden = self.model.init_hidden(1)
            h_out, o_in, o_out, hidden = self.model(src_ohe, hidden, verbose=True)

            h_out_pretty = h_out.view(batch_size, sequence_length, -1)
            o_in_pretty = o_in.view(batch_size, sequence_length, -1)
            o_out_pretty = o_out.view(batch_size, sequence_length, -1)

            W_ih, W_hh, W_ho = self.model.rnn.weight_ih_l0.detach().cpu().numpy(), \
                               self.model.rnn.weight_hh_l0.detach().cpu().numpy(), \
                               self.model.out_linear.weight.detach().cpu().numpy()

            for hidden_out_, out_in_, out_out_, char in zip(h_out_pretty[0],
                                                            o_in_pretty[0],
                                                            o_out_pretty[0],
                                                            word):
                hidden_out_, out_in_, out_out_ = hidden_out_.cpu().numpy(), out_in_.cpu().numpy(), out_out_.cpu().numpy()
                states_log.append(ModelStateLogDTO(word, char,
                                                   hidden_out_, out_in_, out_out_,
                                                   W_ih, W_hh, W_ho,
                                                   self.ix2char, self.char2ix))

        return states_log

