from nn_model import NNModel, ModelStateLogDTO
from experiment_datasets_creator import ExperimentCreator
from phonology_tool import PhonologyTool

train_data_fn = "data/tur_words.txt"
test_data_fn = "data/swadesh.txt"

HIDDEN_SIZE = 2
HIDDEN_TYPE = "sigmoid"
EPOCHS_NUM = 20

train_dataset = []
with open(train_data_fn, 'r', encoding="utf-8") as train_data_f:
    for line in train_data_f:
        train_dataset.append(line.strip())

alphabet = {c for word in train_dataset for c in word}
print(alphabet)


test_dataset = []
with open(test_data_fn, 'r', encoding="utf-8") as test_data_f:
    for line in test_data_f:
        if all(c in alphabet for c in line.strip()):
            test_dataset.append(line.strip())

print(train_dataset)
print(test_dataset)



model = NNModel(alphabet, HIDDEN_SIZE, activation=HIDDEN_TYPE)


train_data = [(entry[:-1], entry[1:]) for entry in train_dataset]
for epoch_num, epoch_loss in model.train(train_data, EPOCHS_NUM):
    print('\t'.join([f"epoch_num: {epoch_num}", f"epoch_loss: {epoch_loss}"]))

pt = PhonologyTool("phon_features.tsv")
ec = ExperimentCreator(model, test_dataset, pt)

print(ec.vov_vs_cons_dataset())


