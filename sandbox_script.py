from use_rnn import load_data
from nn_model import *
import numpy as np


def train_model(filename, hidden_size, activation, num_of_epochs):
    data, alphabet = load_data(filename)
    training_data = [(entry[:-1], entry[1:]) for entry in data]

    model = NNModel(alphabet, hidden_size, activation=activation)
    model.train(training_data, num_of_epochs)

    return model


def process_word(model, word):
    hidden_states = [np.zeros((model.hidden_size, 1))]
    output_activations = []
    predicted_chars = []
    for char in word:
        predicted_chars, units_activations, weights = model.sample_with_logging(char, 1, hidden_out_=hidden_states[-1])
        predicted_chars.append(predicted_chars[0])
        hidden_states.append(units_activations["hidden_layer"][-1])
        output_activations.append(units_activations["output_layer"][-1])

    return hidden_states, output_activations, predicted_chars


# each n-gram will be encoded with ((n+1) * hidden_dim)"h-feature"
# these "h-feature"s are the hidden units activations
# todo: test with (n*vocab_size) "o-feature" things
def word_level_dataset_2_encoded_ngrams_dataset(model, words2target_dataset, ngram_len):
    my_cool_dataset = []
    for (word, feature) in words2target_dataset:
        word_h_features, _, _ = process_word(model=model, word=word)

        current_word_ngrams_features2target = []
        for idx in range(len(word) - ngram_len + 1):
            ngram2h_feature = (word[idx: idx + ngram_len], word_h_features[idx: idx + ngram_len + 1])
            current_word_ngrams_features2target.append(ngram2h_feature)

        for (ngram, ngram_h_features) in current_word_ngrams_features2target:
            my_cool_dataset.append({
                "ngram": ngram,
                "train_features": np.concatenate(ngram_h_features).flatten().tolist(),
                "target_feature": feature
            })

    return my_cool_dataset


def main():
    VOCAB_FILE_FILENAME = r"data/tur_apertium_words.txt"
    EPOCHS_NUM = 1
    HIDDEN_TYPE = "sigmoid"
    HIDDEN_SIZE = 2

    model = train_model(filename=VOCAB_FILE_FILENAME, hidden_size=HIDDEN_SIZE,
                        activation=HIDDEN_TYPE, num_of_epochs=EPOCHS_NUM)

    NGRAM_LEN = 3

    # synthetic testing data. feature: word contains vowel surrounded by consonants
    # TODO: naming ambiguity (phonological) feature != (training) {h,o}_feature
    words2features_dataset = [("asdf", False), ("dollar", True), ("kerel", True)]

    dataset = word_level_dataset_2_encoded_ngrams_dataset(model, words2features_dataset, NGRAM_LEN)
    print(dataset)

if __name__ == "__main__":
    main()
