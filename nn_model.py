import random

import numpy as np

from np_nn_maths import sigmoid, deriv_tanh, deriv_sigmoid, softmax, neg_log


class NNModel:
    def __init__(self, alphabet, hidden_size, activation="sigmoid", learning_rate=0.001, momentum_rate=0.9):

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.HIDDEN_ACTIVATION = activation
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)

        self.char_to_ix = {ch: i for i, ch in enumerate(self.alphabet)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.alphabet)}

        self.W_ih = np.random.randn(self.hidden_size, self.alphabet_size) * 0.01  # input to hidden
        self.W_hh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01  # input to hidden
        self.W_ho = np.random.randn(self.alphabet_size, self.hidden_size) * 0.01  # input to hidden

    def hidden_activation(self, x):
        if self.HIDDEN_ACTIVATION == "tanh":
            return np.tanh(x)
        if self.HIDDEN_ACTIVATION == "sigmoid":
            return sigmoid(x)

    def deriv_hidden_activation(self, activation):
        if self.HIDDEN_ACTIVATION == "tanh":
            return deriv_tanh(activation)
        if self.HIDDEN_ACTIVATION == "sigmoid":
            return deriv_sigmoid(activation)

    def alphabet_position_to_onehot_encode(self, x):
        onehot_encoded = np.zeros((self.alphabet_size, 1))
        onehot_encoded[x] = 1
        return onehot_encoded

    def process_batch(self, xs, ys, hprev):
        input_, hidden_in_, hidden_out_, out_in_, out_out_ = {}, {}, {}, {}, {}

        hidden_out_[-1] = np.copy(hprev)

        # forward pass

        loss = 0
        for t in range(len(xs)):
            x_input = xs[t]
            y_target = ys[t]

            input_[t] = self.alphabet_position_to_onehot_encode(x_input)

            hidden_in_ = self.W_ih @ input_[t] + self.W_hh @ hidden_out_[t - 1]

            hidden_out_[t] = self.hidden_activation(hidden_in_)

            out_in_[t] = self.W_ho @ hidden_out_[t]
            out_out_[t] = softmax(out_in_[t])

            prediction = out_out_[t]

            loss += neg_log(prediction[y_target, 0])

        dW_ih = np.zeros_like(self.W_ih)
        dW_hh = np.zeros_like(self.W_hh)
        dW_ho = np.zeros_like(self.W_ho)

        dh_next = np.zeros((self.hidden_size, 1))
        for t in reversed(range(len(xs))):
            curr_input = input_[t]
            prev_hidden_out = hidden_out_[t - 1]
            curr_hidden_out = hidden_out_[t]
            curr_pred = out_out_[t]
            curr_y_target = ys[t]

            # neg_log_softmax_derivative is pred - y
            dy = np.copy(curr_pred)
            dy[curr_y_target] -= 1

            dW_ho += dy @ curr_hidden_out.T

            dh = self.W_ho.T @ dy + dh_next  # backprop into h
            dhraw = self.deriv_hidden_activation(curr_hidden_out) * dh

            dW_ih += dhraw @ curr_input.T  # derivative of input to hidden layer weight
            dW_hh += dhraw @ prev_hidden_out.T  # derivative of hidden layer to hidden layer weight
            dh_next = self.W_hh.T @ dhraw

        last_hidden_out = hidden_out_[len(xs) - 1]
        return loss, dW_ih, dW_hh, dW_ho, last_hidden_out

    # prediction, one full forward pass
    def sample(self, hidden_out_, seed_idx, n):
        predicted_char_onehot = self.alphabet_position_to_onehot_encode(seed_idx)

        predicted_chars_sequence = []
        predicted_chars_indices_sequence = []

        for t in range(n):
            hidden_out_ = self.hidden_activation(self.W_ih @ predicted_char_onehot + self.W_hh @ hidden_out_)  # + bh)
            out_in_ = self.W_ho @ hidden_out_  # + by
            out_out_ = softmax(out_in_)

            out_out_flattened = out_out_.ravel()

            predicted_char_alphabet_idx = np.random.choice(range(self.alphabet_size), p=out_out_flattened)
            predicted_char_onehot = self.alphabet_position_to_onehot_encode(predicted_char_alphabet_idx)
            predicted_chars_sequence.append(self.ix_to_char[predicted_char_alphabet_idx])

            # actually not used
            predicted_chars_indices_sequence.append(predicted_char_alphabet_idx)

        predicted_string = ''.join(predicted_chars_sequence)
        return predicted_string

    # prediction, one full forward pass
    def sample_with_logging(self, seed_char, n, hidden_out_=None):

        if hidden_out_ is None:
            hidden_out_ = np.zeros((self.hidden_size, 1))

        seed_idx = self.char_to_ix[seed_char]
        predicted_char_onehot = self.alphabet_position_to_onehot_encode(seed_idx)

        predicted_chars_sequence = []
        # predicted_chars_indices_sequence = []

        hidden_outs = []
        output_outs = []
        for t in range(n):
            hidden_out_ = self.hidden_activation(self.W_ih @ predicted_char_onehot + self.W_hh @ hidden_out_)  # + bh)
            out_in_ = self.W_ho @ hidden_out_  # + by
            out_out_ = softmax(out_in_)

            out_out_flattened = out_out_.ravel()

            predicted_char_alphabet_idx = np.random.choice(range(self.alphabet_size), p=out_out_flattened)  # todo fix
            predicted_char_onehot = self.alphabet_position_to_onehot_encode(predicted_char_alphabet_idx)
            predicted_chars_sequence.append(self.ix_to_char[predicted_char_alphabet_idx])

            hidden_outs.append(hidden_out_)
            output_outs.append(out_out_)

        units_activations = {"hidden_layer": hidden_outs, "output_layer": output_outs}
        weights = {"IH": self.W_ih, "HH": self.W_hh, "HO": self.W_ho}
        return predicted_chars_sequence, units_activations, weights

    def train(self, training_data, num_of_epochs, logging=True, sample=False):
        EPOCHS_NUM = num_of_epochs

        momentum_delta_W_ih = np.zeros_like(self.W_ih)
        momentum_delta_W_hh = np.zeros_like(self.W_hh)
        momentum_delta_W_ho = np.zeros_like(self.W_ho)
        h_prev = np.zeros((self.hidden_size, 1))

        for epoch_num in range(EPOCHS_NUM):
            for batch in training_data:
                h_prev = np.zeros((self.hidden_size, 1))  # reset RNN memory

                inputs = [self.char_to_ix[ch] for ch in batch[0]]
                targets = [self.char_to_ix[ch] for ch in batch[1]]

                epoch_loss, dW_ih, dW_hh, dW_ho, h_prev = self.process_batch(inputs, targets, h_prev)

                momentum_delta_W_ih = self.learning_rate * dW_ih + self.momentum_rate * momentum_delta_W_ih
                self.W_ih -= momentum_delta_W_ih

                momentum_delta_W_hh = self.learning_rate * dW_hh + self.momentum_rate * momentum_delta_W_hh
                self.W_hh -= momentum_delta_W_hh

                momentum_delta_W_ho = self.learning_rate * dW_ho + self.momentum_rate * momentum_delta_W_ho
                self.W_ho -= momentum_delta_W_ho

            if logging:
                print('iter %d, loss: %f' % (epoch_num, epoch_loss))  # print progress
                if sample:
                    h_prev = np.zeros((self.hidden_size, 1))
                    sample_starting_char = random.choice(list(self.char_to_ix.keys()))
                    keys_ = self.char_to_ix[sample_starting_char]
                    print("sample: ", sample_starting_char+self.sample(h_prev, keys_, 10))