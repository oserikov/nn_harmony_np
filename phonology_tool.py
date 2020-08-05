from collections import defaultdict as dd


class PhonologyTool:

    def __init__(self, phonology_features_filename):
        self.char2features = dd(set)
        with open(phonology_features_filename, 'r', encoding="utf-8") as features_f:
            for line in features_f:
                letter, features = line.strip().split('\t', 1)
                self.char2features[letter].update(features.split())

    def is_vowel(self, char):
        return "vow" in self.char2features.get(char, [])

    def is_consonant(self, char):
        return not self.is_vowel(char)

    def is_front(self, char):
        return self.is_vowel(char) and "+front" in self.char2features.get(char, [])

    def is_back(self, char):
        return self.is_vowel(char) and "-front" in self.char2features.get(char, [])

    def is_round(self, char):
        return self.is_vowel(char) and "+round" in self.char2features.get(char, [])

    def is_voiced_stop_consonant(self, char):
        return self.is_consonant(char) \
               and "+voiced" in self.char2features.get(char, []) \
               and "+stop" in self.char2features.get(char, [])

    def shows_front_harmony(self, string):
        res = all([self.is_front(c) for c in string if self.is_vowel(c)]) \
              and len([c for c in string if self.is_vowel(c)]) > 1
        return res

    def shows_back_harmony(self, string):
        return all([self.is_back(c) for c in string if self.is_vowel(c)])
