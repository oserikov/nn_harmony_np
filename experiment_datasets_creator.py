import numpy as np

from phonology_tool import PhonologyTool
from nn_model import NNModel, ModelStateLogDTO
from typing import List, Callable


# 1. vow vs cons
# 2. +front vs -front
# 3. voiced stop consonant detection
# 4. start cons cluster
# 5. 2nd in cons cluster
# 6. +front harmony TODO
# 7. -front_harmony TODO


class ExperimentCreator:
    def __init__(self, nn_model: NNModel, dataset: List[str], phonology_tool: PhonologyTool):
        self.nn_model = nn_model
        self.dataset = dataset
        self.phon_tool = phonology_tool

    def construct_unigram_dataset(self, char2target_fun: Callable, nn_feature_extractor_fun: Callable):
        training_data = []
        for word in self.dataset:
            nn_features = self.get_nn_features_for_word(word)

            word_training_data = []
            for nn_feature in nn_features:
                word_training_data.append((nn_feature_extractor_fun(nn_feature), char2target_fun(nn_feature.char)))

            training_data.extend(word_training_data)

        return training_data

    def vov_vs_cons_dataset(self):
        return self.construct_unigram_dataset(self.phon_tool.is_vowel, self.extract_all_nn_features)

    def front_feature_dataset(self):
        return self.construct_unigram_dataset(self.phon_tool.is_front, self.extract_all_nn_features)

    def voiced_stop_consonant_dataset(self):
        return self.construct_unigram_dataset(self.phon_tool.is_voiced_stop_consonant, self.extract_all_nn_features)

    def second_consonant_in_cluster_dataset(self):

        nn_feature_extractor_fun = self.extract_all_nn_features

        training_data = []
        for word in self.dataset:
            nn_features = self.get_nn_features_for_word(word)
            nn_feature = nn_features[0]

            word_training_data = [(nn_feature_extractor_fun(nn_feature), False)]

            previous_is_first_consonant_in_cluster = self.phon_tool.is_consonant(nn_feature.char)
            previous_is_vowel = self.phon_tool.is_vowel(nn_feature.char)
            for nn_feature in nn_features:
                word_training_data.append((nn_feature_extractor_fun(nn_feature),
                                           self.phon_tool.is_consonant(nn_feature.char)
                                           and previous_is_first_consonant_in_cluster))

                previous_is_first_consonant_in_cluster = self.phon_tool.is_consonant(nn_feature.char) \
                                                         and previous_is_vowel
                previous_is_vowel = self.phon_tool.is_vowel(nn_feature.char)

            training_data.extend(word_training_data)

        return training_data

    def is_starting_consonant_cluster_dataset(self):

        nn_feature_extractor_fun = self.extract_all_nn_features
        training_data = []

        for word in self.dataset:
            nn_features = self.get_nn_features_for_word(word)
            nn_feature = nn_features[0]

            word_training_data = [(nn_feature_extractor_fun(nn_feature), self.phon_tool.is_consonant(nn_feature.char))]

            previous_is_vowel = self.phon_tool.is_vowel(nn_feature.char)
            for nn_feature in nn_features:
                word_training_data.append((nn_feature_extractor_fun(nn_feature),
                                           self.phon_tool.is_consonant(nn_feature.char) and previous_is_vowel))

                previous_is_vowel = self.phon_tool.is_vowel(nn_feature.char)

            training_data.extend(word_training_data)

        return training_data

    def extract_all_nn_features(self, nn_feature: ModelStateLogDTO):
        # res =
        # for k,v in res.items():
        #     res[k] = np.ravel(v).tolist()
        return vars(nn_feature)

    def get_nn_features_for_word(self, word) -> List[ModelStateLogDTO]:
        return self.nn_model.run_model_on_word(word)
