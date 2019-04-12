from nn_model import NNModel, ModelStateLogDTO
from experiment_datasets_creator import ExperimentCreator
from phonology_tool import PhonologyTool

train_data_fn = "data/tur_apertium_words.txt"
test_data_fn = "data/tur_swadesh.txt"

HIDDEN_SIZE = 2
HIDDEN_TYPE = "sigmoid"
EPOCHS_NUM = 2
MODEL_FILENAME = "tmp_model_fn"

train_dataset = []
with open(train_data_fn, 'r', encoding="utf-8") as train_data_f:
    for line in train_data_f:
        train_dataset.append(line.strip())

alphabet = {c for word in train_dataset for c in word}
print(alphabet)


print(train_dataset)


model = NNModel(alphabet, HIDDEN_SIZE, activation=HIDDEN_TYPE)

train_data = [(entry[:-1], entry[1:]) for entry in train_dataset]
for epoch_num, epoch_loss in model.train(train_data, EPOCHS_NUM):
    print('\t'.join([f"epoch_num: {epoch_num}", f"epoch_loss: {epoch_loss}"]))



model.save(MODEL_FILENAME)

del model

model = NNModel.load_model(MODEL_FILENAME)


test_dataset = []
with open(test_data_fn, 'r', encoding="utf-8") as test_data_f:
    for line in test_data_f:
        if all(c in alphabet for c in line.strip()):
            test_dataset.append(line.strip())
print(test_dataset)
pt = PhonologyTool("tur_phon_features.tsv")
ec = ExperimentCreator(model, test_dataset, pt)

dataset = ec.front_harmony_dataset()


def train_to_single_dict(train_dicts_list):
    res = {}
    for idx, d in enumerate(train_dicts_list):
        for k, v in d.items():
            res[f"{idx}_" + k] = v
    return res


pretty_dataset = []
for (train_entry, target_entry) in dataset:
    pretty_dataset.append((train_to_single_dict(train_entry), target_entry))

first_entry_keys = list(pretty_dataset[0][0].keys())
first_entry_keys_not_unique = set(first_entry_keys[3:])

good = [entry for entry in pretty_dataset if entry[1]]

bad = [entry for entry in pretty_dataset if not entry[1]]


