import numpy as np

from log_reader import LogReader
import matplotlib
from matplotlib import pyplot as plt

import sys

DEBUG = True

l_reader = LogReader()
l_reader.parse_log_file(sys.stdin)

weight_log = l_reader.weight_log
activations_log = l_reader.activations_log
iter_loss_log = l_reader.iter_loss_log

if DEBUG:
    print("weight_log", weight_log)
    print("activations_log", activations_log)
    print("iter_loss_log", iter_loss_log)

ORDERED_ALPHABET = "abcçdefgğhıijklmnoöprsştuüvyz"
vowels = [c for c in "aeiıoöüu"]

letters_to_latex_encoded_letters = {letter_name: r'$\mathrm{' + letter_name + '}$' for letter_name in ORDERED_ALPHABET}

# plot hidden units activatinos

all_activations = []
for activations_of_unit in activations_log["HIDDEN"].values():
    for activation_for_char in activations_of_unit.values():
        all_activations.append(activation_for_char["value"])
lowest_activation_value = np.floor(np.min(all_activations)) - 0.1
highest_activation_value = np.ceil(np.max(all_activations)) + 0.1


matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath},\usepackage{amssymb}')
for unit_idx, unit_activations_for_letters in activations_log["HIDDEN"].items():
    ax = plt.subplot(111)
    ax.set_ylim([lowest_activation_value, highest_activation_value])

    for letter_idx, (letter_name, letter_in_latex) in enumerate(letters_to_latex_encoded_letters.items()):
        value = unit_activations_for_letters[letter_name]["value"]

        if letter_name in vowels:
            ax.plot(letter_idx, value, "o", mfc='none', color='C0')
        else:
            ax.plot(letter_idx, value, "o", color='C0')
        ax.annotate(letter_in_latex, xy=(letter_idx - 0.3, value - 0.06))

    ax.set_yticks((0, 1))
    ax.set_xticks(())

    plt.savefig('unit_' + str(unit_idx) + '.png')

    plt.clf()
