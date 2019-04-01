import random

import numpy as np

# model parameters
hidden_size = 100
learning_rate = 0.001
momentum_rate = 0.9

HIDDEN_ACTIVATION = "tanh"
# HIDDEN_ACTIVATION = "sigmoid"

data = []
alphabet = set()
with open('tur_words.txt', 'r', encoding="utf-8") as f:
    for line in f:
        line = line.rstrip()
        if not line:
            continue
        # entry = line
        for word in line.split():
            entry = word
            data.append(entry)
            alphabet.update({char for char in entry})

alphabet_size = len(alphabet)

char_to_ix = {ch: i for i, ch in enumerate(alphabet)}
ix_to_char = {i: ch for i, ch in enumerate(alphabet)}



W_ih = np.random.randn(hidden_size, alphabet_size) * 0.01  # input to hidden
W_hh = np.random.randn(hidden_size, hidden_size) * 0.01  # input to hidden
W_ho = np.random.randn(alphabet_size, hidden_size) * 0.01  # input to hidden


def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=0)).T
    return sm


def deriv_tanh(tanh_value):
    return 1 - tanh_value * tanh_value


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def deriv_sigmoid(sigmoid):
    return sigmoid * (1. - sigmoid)


def onehot_alphabet_encode(x):
    onehot_encoded = np.zeros((alphabet_size, 1))
    onehot_encoded[x] = 1
    return onehot_encoded


def neg_log(x):
    # TODO: does this actually work?
    return -np.log(x)


def hidden_activation(x):
    if HIDDEN_ACTIVATION == "tanh":
        return np.tanh(x)
    if HIDDEN_ACTIVATION == "sigmoid":
        return sigmoid(x)

def deriv_hidden_activation(activation):
    if HIDDEN_ACTIVATION == "tanh":
        return deriv_tanh(activation)
    if HIDDEN_ACTIVATION == "sigmoid":
        return deriv_sigmoid(activation)

def process_batch(xs, ys, hprev):

    input_, hidden_in_, hidden_out_, out_in_, out_out_ = {}, {}, {}, {}, {}

    hidden_out_[-1] = np.copy(hprev)

    # forward pass

    loss = 0
    for t in range(len(xs)):
        x_input = xs[t]
        y_target = ys[t]

        input_[t] = onehot_alphabet_encode(x_input)

        hidden_in_ = W_ih @ input_[t] + W_hh @ hidden_out_[t - 1]

        hidden_out_[t] = hidden_activation(hidden_in_)

        out_in_[t] = W_ho @ hidden_out_[t]
        out_out_[t] = softmax(out_in_[t])

        prediction = out_out_[t]

        loss += neg_log(prediction[y_target, 0])

    dW_ih = np.zeros_like(W_ih)
    dW_hh = np.zeros_like(W_hh)
    dW_ho = np.zeros_like(W_ho)

    dh_next = np.zeros((hidden_size, 1))
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

        dh = W_ho.T @ dy + dh_next  # backprop into h
        dhraw = deriv_hidden_activation(curr_hidden_out) * dh

        dW_ih += dhraw @ curr_input.T  # derivative of input to hidden layer weight
        dW_hh += dhraw @ prev_hidden_out.T  # derivative of hidden layer to hidden layer weight
        dh_next = W_hh.T @ dhraw

    last_hidden_out = hidden_out_[len(xs) - 1]
    return loss, dW_ih, dW_hh, dW_ho, last_hidden_out


# prediction, one full forward pass
def sample(hidden_out_, seed_ix, n):
    predicted_char_onehot = onehot_alphabet_encode(seed_ix)

    # list to store generated chars
    predicted_chars_indices_sequence = []

    predicted_chars_sequence = []

    # for as many characters as we want to generate
    for t in range(n):
        hidden_out_ = hidden_activation(W_ih @ predicted_char_onehot + W_hh @ hidden_out_)  # + bh)
        out_in_ = W_ho @ hidden_out_  # + by
        out_out_ = softmax(out_in_)

        out_out_flattened = out_out_.ravel()

        predicted_char_alphabet_idx = np.random.choice(range(alphabet_size), p=out_out_flattened)
        predicted_char_onehot = onehot_alphabet_encode(predicted_char_alphabet_idx)
        predicted_chars_sequence.append(ix_to_char[predicted_char_alphabet_idx])

        # actually not used
        predicted_chars_indices_sequence.append(predicted_char_alphabet_idx)

    predicted_string = ''.join(predicted_chars_sequence)
    print('----\n %s \n----' % (predicted_string,))

#
# def sample_with_logging(hidden_out_, seed_ix):
#     predicted_char_onehot = onehot_alphabet_encode(seed_ix)
#
#     # list to store generated chars
#     predicted_chars_indices_sequence = []
#
#     predicted_chars_sequence = []
#
#     # for as many characters as we want to generate
#
#         hidden_out_ = hidden_activation(W_ih @ predicted_char_onehot + W_hh @ hidden_out_)  # + bh)
#         out_in_ = W_ho @ hidden_out_  # + by
#         out_out_ = softmax(out_in_)
#
#         out_out_flattened = out_out_.ravel()
#
#         predicted_char_alphabet_idx = np.random.choice(range(alphabet_size), p=out_out_flattened)
#         predicted_char_onehot = onehot_alphabet_encode(predicted_char_alphabet_idx)
#         predicted_chars_sequence.append(ix_to_char[predicted_char_alphabet_idx])
#
#         # actually not used
#         predicted_chars_indices_sequence.append(predicted_char_alphabet_idx)
#
#     predicted_string = ''.join(predicted_chars_sequence)
#     print('----\n %s \n----' % (predicted_string,))



EPOCHS_NUM = 100


training_data = [(entry[:-1], entry[1:]) for entry in data]


momentum_delta_W_ih = np.zeros_like(W_ih)
momentum_delta_W_hh = np.zeros_like(W_hh)
momentum_delta_W_ho = np.zeros_like(W_ho)
h_prev = np.zeros((hidden_size, 1))

for epoch_num in range(EPOCHS_NUM):
    for batch in training_data:
        h_prev = np.zeros((hidden_size, 1))  # reset RNN memory

        inputs = [char_to_ix[ch] for ch in batch[0]]
        targets = [char_to_ix[ch] for ch in batch[1]]

        epoch_loss, dW_ih, dW_hh, dW_ho, h_prev = process_batch(inputs, targets, h_prev)

        momentum_delta_W_ih = learning_rate * dW_ih + momentum_rate * momentum_delta_W_ih
        W_ih -= momentum_delta_W_ih

        momentum_delta_W_hh = learning_rate * dW_hh + momentum_rate * momentum_delta_W_hh
        W_hh -= momentum_delta_W_hh

        momentum_delta_W_ho = learning_rate * dW_ho + momentum_rate * momentum_delta_W_ho
        W_ho -= momentum_delta_W_ho


    print('iter %d, loss: %f' % (epoch_num, epoch_loss))  # print progress
    h_prev = np.zeros((hidden_size, 1))
    keys_ = char_to_ix[random.choice(list(char_to_ix.keys()))]
    sample(h_prev, keys_, 10)
    # sample_with_logging(h_prev, keys_)
