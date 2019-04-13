import numpy as np

from phonology_tool import PhonologyTool
from nn_model import NNModel, ModelStateLogDTO
from typing import List, Callable
import csv

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

    def construct_ngram_dataset(self, ngram2target_fun: Callable, nn_feature_extractor_fun: Callable, ngram_len: int):
        training_data = []
        for word in self.dataset:
            nn_features = self.get_nn_features_for_word(word)

            word_training_data = []
            for idx, nn_feature in enumerate(nn_features[:-ngram_len + 1]):
                ngram = ''.join([f.char for f in nn_features[idx:idx+ngram_len]])
                tmp_features = [{"ngram": ngram}]
                for idx, tmp_feature in enumerate(nn_features[idx:idx+ngram_len]):
                    tmp_features.append({f"char_{idx}_"+k: v
                                         for entry in nn_feature_extractor_fun(tmp_feature)
                                         for k, v in entry.items()})
                word_training_data.append((tmp_features, ngram2target_fun(ngram)))
                # word_training_data.append((nn_feature_extractor_fun(nn_feature), ngram2target_fun(nn_feature.char)))

            training_data.extend(word_training_data)

        return training_data

    def front_harmony_dataset(self):
        return self.construct_ngram_dataset(self.phon_tool.shows_front_harmony, self.extract_all_nn_features, 4)

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
        return nn_feature.as_dict()

    def get_nn_features_for_word(self, word) -> List[ModelStateLogDTO]:
        return self.nn_model.run_model_on_word(word)


    @staticmethod
    def train_eatures_to_single_dict(dataset_train_features_list):
        res = {}
        for idx, d in enumerate(dataset_train_features_list):
            for k, v in d.items():
                res[f"{idx}_" + k] = v
        return res


    @staticmethod
    def make_dataset_pretty(dataset):
        pretty_dataset = []
        for (train_entry, target_entry) in dataset:
            pretty_dataset.append((ExperimentCreator.train_eatures_to_single_dict(train_entry), target_entry))
        return pretty_dataset


    @staticmethod
    def save_dataset_to_tsv(dataset, dataset_fn):

        dataset_to_single_dicts_list = []
        for entry in dataset:
            new_entry = entry[0].copy()
            new_entry["TARGET"] = entry[1]
            dataset_to_single_dicts_list.append(new_entry)

        dataset_keys = list(dataset_to_single_dicts_list[0].keys())
        with open(dataset_fn, 'w', encoding="utf-8", newline='') as dataset_f:
            dictWriter = csv.DictWriter(dataset_f, dataset_to_single_dicts_list[0].keys(), delimiter='\t')
            dictWriter.writeheader()
            dictWriter.writerows(dataset_to_single_dicts_list)

            # print('\t'.join(dataset_keys + ["target"]), file=dataset_f)
            # for (features, target) in dataset:
            #     print('\t'.join([features[key] for key in dataset_keys] + [target]), file=dataset_f)

