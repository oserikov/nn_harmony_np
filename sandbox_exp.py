from nn_model import NNModel, ModelStateLogDTO
from experiment_datasets_creator import ExperimentCreator
from phonology_tool import PhonologyTool

MODEL_FILENAME = "../model_size_2_activation_sigmoid.pkl"
test_data_fn = "data/tur_swadesh.txt"
phonology_features_filename = "data/tur_phon_features.tsv"

model = NNModel.load_model(MODEL_FILENAME)

test_dataset = []
with open(test_data_fn, 'r', encoding="utf-8") as test_data_f:
    for line in test_data_f:
        if all(c in model.alphabet for c in line.strip()):
            test_dataset.append(line.strip())

phonologyTool = PhonologyTool(phonology_features_filename)
experimentCreator = ExperimentCreator(model, test_dataset, phonologyTool)

# front_harmony_dataset
front_harmony_dataset_fn = "front_harmony_dataset.tsv"
front_harmony_dataset = experimentCreator.make_dataset_pretty(experimentCreator.front_harmony_dataset())
experimentCreator.save_dataset_to_tsv(front_harmony_dataset, front_harmony_dataset_fn)

# vov_vs_cons_dataset
vov_vs_cons_dataset_fn = "vov_vs_cons_dataset.tsv"
vov_vs_cons_dataset = experimentCreator.make_dataset_pretty(experimentCreator.vov_vs_cons_dataset())
experimentCreator.save_dataset_to_tsv(vov_vs_cons_dataset, vov_vs_cons_dataset_fn)

# front_feature_dataset
front_feature_dataset_fn = "front_feature_dataset.tsv"
front_feature_dataset = experimentCreator.make_dataset_pretty(experimentCreator.front_feature_dataset())
experimentCreator.save_dataset_to_tsv(front_feature_dataset, front_feature_dataset_fn)

# is_starting_consonant_cluster_dataset
is_starting_consonant_cluster_dataset_fn = "is_starting_consonant_cluster_dataset.tsv"
is_starting_consonant_cluster_dataset = experimentCreator.make_dataset_pretty(experimentCreator.is_starting_consonant_cluster_dataset())
experimentCreator.save_dataset_to_tsv(is_starting_consonant_cluster_dataset, is_starting_consonant_cluster_dataset_fn)

# second_consonant_in_cluster_dataset
second_consonant_in_cluster_dataset_fn = "second_consonant_in_cluster_dataset.tsv"
second_consonant_in_cluster_dataset = experimentCreator.make_dataset_pretty(experimentCreator.second_consonant_in_cluster_dataset())
experimentCreator.save_dataset_to_tsv(second_consonant_in_cluster_dataset, second_consonant_in_cluster_dataset_fn)

# voiced_stop_consonant_dataset
voiced_stop_consonant_dataset_fn = "voiced_stop_consonant_dataset.tsv"
voiced_stop_consonant_dataset = experimentCreator.make_dataset_pretty(experimentCreator.voiced_stop_consonant_dataset())
experimentCreator.save_dataset_to_tsv(voiced_stop_consonant_dataset, voiced_stop_consonant_dataset_fn)




