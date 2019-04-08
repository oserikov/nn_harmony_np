from nn_model import NNModel
import sys
import traceback

EPOCHS_NUM = 100
HIDDEN_SIZE = 2

# HIDDEN_TYPE="tanh"
HIDDEN_TYPE = "sigmoid"


# HIDDEN_TYPE = "relu"


def help_and_exit():
    print("usage: python3 THIS_SCRIPT.py HIDDEN_SIZE EPOCHS_NUM FILENAME")
    print("HIDDEN_SIZE is the integer number of units in the hidden layer")
    print("EPOCHS_NUM  is the number of training epochs")
    print("FILENAME (default: STDIN) is the relative path to vocabulary file to train on.\n" +
          "  pass STDIN as FILENAME to use stdin as the vocabulary file")
    exit(0)


def load_data(vocab_fn):
    loaded_data = []
    loaded_alphabet = set()

    if vocab_fn == "STDIN":
        f = sys.stdin
    else:
        f = open(vocab_fn, 'r', encoding="utf-8")

    for line in f:
        line = line.rstrip()

        try:
            assert len(line.split()) == 1
        except AssertionError:
            print("the data should be a word per line file. error on line: " + line)
            exit(1)

        line_word = line.split()[0]
        loaded_data.append(line_word)
        loaded_alphabet.update({c for c in line_word})

    if vocab_fn != "STDIN":
        f.close()

    return loaded_data, loaded_alphabet


def main():
    if len(sys.argv) > 1 and any(["--help" in argv for argv in sys.argv[1:]]):
        help_and_exit()

    VOCAB_FILE_FILENAME = "STDIN"
    try:
        HIDDEN_SIZE = int(sys.argv[1])
        EPOCHS_NUM = int(sys.argv[2])
        if len(sys.argv) > 3:
            VOCAB_FILE_FILENAME = sys.argv[3]
    except Exception:
        traceback.print_exc()
        print()
        help_and_exit()

    data, alphabet = load_data(VOCAB_FILE_FILENAME)
    training_data = [(entry[:-1], entry[1:]) for entry in data]

    model = NNModel(alphabet, HIDDEN_SIZE, activation=HIDDEN_TYPE)
    model.train(training_data, EPOCHS_NUM)

    hidden_unit_activation_for_char = [{} for unit_idx in range(HIDDEN_SIZE)]
    output_unit_activation_for_char = [{} for unit_idx in range(len(alphabet))]

    for char in alphabet:
        predicted_chars, units_activations, weights = model.sample_with_logging(char, 1)
        predicted_char = predicted_chars[0]
        hidden_activations = units_activations["hidden_layer"]
        output_activations = units_activations["output_layer"]

        hidden_units_activations = [h_u_act[0] for h_u_act in hidden_activations[0]]
        output_units_activations = [o_u_act[0] for o_u_act in output_activations[0]]

        for unit_idx, unit_activation in enumerate(hidden_units_activations):
            hidden_unit_activation_for_char[unit_idx][char] = hidden_units_activations[unit_idx]

        for unit_idx, unit_activation in enumerate(output_units_activations):
            output_unit_activation_for_char[unit_idx][char] = output_units_activations[unit_idx]

    for unit_idx, unit_activations in enumerate(hidden_unit_activation_for_char):
        for char in alphabet:
            print(f"activation of HIDDEN unit {unit_idx} for char {char}" + '\t' +
                  str(hidden_unit_activation_for_char[unit_idx][char]))

    for unit_idx, unit_activations in enumerate(output_unit_activation_for_char):
        for char in alphabet:
            output_char = model.ix_to_char[unit_idx]
            print(f"activation of OUTPUT unit {unit_idx} (represents char {output_char}) for char {char}" + '\t' +
                  str(output_unit_activation_for_char[unit_idx][char]))

    for hidden_idx, from_input_to_idxth_hidden in enumerate(model.W_ih):
        for char_idx, weight in enumerate(from_input_to_idxth_hidden):
            input_char = model.ix_to_char[char_idx]
            print(f"weight INPUT unit {char_idx} (represents char {input_char}) to HIDDEN unit {hidden_idx}" + '\t' +
                  str(weight))

    for hidden_tgt_idx, from_hidden_to_idxth_hidden in enumerate(model.W_hh):
        for hidden_src_idx, weight in enumerate(from_hidden_to_idxth_hidden):
            print(f"weight HIDDEN unit {hidden_src_idx} to HIDDEN unit {hidden_tgt_idx}" + '\t' + str(weight))

    for output_idx, from_hidden_to_idxth_output in enumerate(model.W_ho):
        for hidden_idx, weight in enumerate(from_hidden_to_idxth_output):
            output_char = model.ix_to_char[output_idx]
            print(
                f"weight HIDDEN unit {hidden_idx} to OUTPUT unit {output_idx} (represents char {output_char})" + '\t' +
                str(weight))


if __name__ == "__main__":
    main()