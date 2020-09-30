import sys
# sys.path.append(os.path.join(os.getcwd(), "nn_harmony_np"))
import os
import unicodedata
from collections import defaultdict
import subprocess
import pandas as pd
import torch
from livelossplot import PlotLosses
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pickle
from experiment_datasets_creator import ExperimentCreator
from model_runner import ModelRunner
from phonology_tool import PhonologyTool
from torch_rnns import *
from torch_train_test import train, test
from tree2pseudo import dt_probe_dataset
# from google.colab import files
from worddata import WordData
import json


HIDDEN_SIZE = 3
LANG = "tur"
if len(sys.argv) > 2:
    LANG = sys.argv[1]
    HIDDEN_SIZE = int(sys.argv[2])

print(f"lang: {LANG}\th_size: {HIDDEN_SIZE}")

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


tur_df = pd.read_csv(
    f"https://raw.githubusercontent.com/oserikov/nn_harmony_np/master/data/{LANG}/{LANG}_words_clean_syllabified.txt",
    delimiter=' ', names=["wf", "syllabified"])

tur_df["syllabified"] = tur_df["syllabified"].apply(strip_accents)
tur_df["src"] = tur_df["wf"].apply(lambda x: x[:-1])
tur_df["tgt"] = tur_df["wf"].apply(lambda x: x[1:])
tur_df = tur_df[["src", "tgt"]]

def initialize_smth2index():
    smth2index = defaultdict(lambda: len(smth2index))
    return smth2index

char2index = initialize_smth2index()
for lem in list(tur_df.src):
    for char in lem:
        char2index[char]

for wf in list(tur_df.tgt):
    for char in wf:
        char2index[char]

char2index[WordData.WF_PAD_TOKEN]
char2index[WordData.WF_EOS_TOKEN]


_train_df, test_df = train_test_split(tur_df, test_size=0.1)
train_df, val_df = train_test_split(_train_df, test_size=0.2)

src_test, tgt_test = test_df.src, test_df.tgt
src_train, tgt_train = train_df.src, train_df.tgt
src_val, tgt_val = val_df.src, val_df.tgt

BATCH_SIZE = 64

train_dataset = WordData(list(src_train), list(tgt_train), char2index)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

validation_dataset = WordData(list(src_val), list(tgt_val), char2index)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)

test_dataset = WordData(list(src_test), list(tgt_test), char2index)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


def build_datasets(model_runner, model_filename="model_size_6_activation_tanh"):
    orig_data_fn = f"data/{LANG}/{LANG}_words_clean_syllabified.txt"
    phonology_features_filename = f"data/{LANG}/{LANG}_features.tsv"
    test_data_fn = "probing_subset.txt"

    pd.read_csv(orig_data_fn, names=["w", "s"], sep=' ')["w"].sample(n=250).to_csv(test_data_fn, header=False, index=False)

    datasets = []
    # for model_filename in ["model_size_6_activation_tanh"]:
    model_filename_prefix = model_filename.rstrip(".pkl")

    test_dataset = []
    with open(test_data_fn, 'r', encoding="utf-8") as test_data_f:
        for line in test_data_f:
            if all(c in model_runner.char2ix.keys() for c in line.strip()):
                test_dataset.append(line.strip())

    phonologyTool = PhonologyTool(phonology_features_filename)
    experimentCreator = ExperimentCreator(model_runner, test_dataset, phonologyTool)



    if LANG == "tur":
        # front_feature_dataset
        front_feature_dataset_fn = model_filename_prefix + "_front_feature_dataset.tsv"
        front_feature_dataset = experimentCreator.make_dataset_pretty(experimentCreator.front_feature_dataset())
        front_feature_dataset = [e for e in front_feature_dataset if phonologyTool.is_vowel(e[0]["1_char"])]
        experimentCreator.save_dataset_to_tsv(front_feature_dataset, front_feature_dataset_fn)
        datasets.append(front_feature_dataset_fn)

        # round_feature_dataset
        round_feature_dataset_fn = model_filename_prefix + "_round_feature_dataset.tsv"
        round_feature_dataset = experimentCreator.make_dataset_pretty(experimentCreator.round_feature_dataset())
        round_feature_dataset = [e for e in round_feature_dataset if phonologyTool.is_vowel(e[0]["1_char"])]
        experimentCreator.save_dataset_to_tsv(round_feature_dataset, round_feature_dataset_fn)
        datasets.append(round_feature_dataset_fn)

    elif LANG == "rus":
        # front_feature_dataset
        front_feature_dataset_fn = model_filename_prefix + "_front_feature_dataset.tsv"
        front_feature_dataset = experimentCreator.make_dataset_pretty(experimentCreator.front_feature_dataset())
        front_feature_dataset = [e for e in front_feature_dataset if phonologyTool.is_vowel(e[0]["1_char"])]
        experimentCreator.save_dataset_to_tsv(front_feature_dataset, front_feature_dataset_fn)
        datasets.append(front_feature_dataset_fn)

    elif LANG == "swa":
        class wHeightPhonTool(PhonologyTool):
            def is_high(self, char):
                res = self.is_vowel(char) and "+high" in self.char2features.get(char, [])
                return res

        class wHeightExp(ExperimentCreator):
            def height_feature_dataset(self):
                return self.construct_unigram_dataset(self.phon_tool.is_high, self.extract_all_nn_features)

        phonologyTool = wHeightPhonTool(phonology_features_filename)
        experimentCreator = wHeightExp(model_runner, test_dataset, phonologyTool)

        # height_feature_dataset_small
        height_feature_dataset_small_fn = model_filename_prefix + "_height_feature_small_dataset.tsv"
        height_feature_dataset_small = experimentCreator.make_dataset_pretty(experimentCreator.height_feature_dataset())

        height_feature_dataset_small = [e for e in height_feature_dataset_small
                                        if e[0]["1_char"] in "eiou"]
        experimentCreator.save_dataset_to_tsv(height_feature_dataset_small, height_feature_dataset_small_fn)
        datasets.append(height_feature_dataset_small_fn)

    elif LANG == "fin":
        # front_feature_dataset
        front_feature_dataset_fn = model_filename_prefix + "_front_feature_dataset.tsv"
        front_feature_dataset = experimentCreator.make_dataset_pretty(experimentCreator.front_feature_dataset())
        front_feature_dataset = [e for e in front_feature_dataset if phonologyTool.is_vowel(e[0]["1_char"])]
        experimentCreator.save_dataset_to_tsv(front_feature_dataset, front_feature_dataset_fn)
        datasets.append(front_feature_dataset_fn)

    vov_vs_cons_dataset_fn = model_filename_prefix + "_vov_vs_cons_dataset.tsv"
    vov_vs_cons_dataset = experimentCreator.make_dataset_pretty(experimentCreator.vov_vs_cons_dataset())
    experimentCreator.save_dataset_to_tsv(vov_vs_cons_dataset, vov_vs_cons_dataset_fn)
    datasets.append(vov_vs_cons_dataset_fn)

    return datasets


def build_sequential_datasets(model_runner, model_filename="model_size_6_activation_tanh"):
    test_data_fn = f"data/{LANG}/{LANG}_words_clean_syllabified.txt"
    phonology_features_filename = f"data/{LANG}/{LANG}_features.tsv"

    os.system(f"shuf < (cut - d' ' -f1 {test_data_fn}) | head - 250 > probing_subset.txt")
    test_data_fn = "probing_subset.txt"

    datasets = []
    model_filename_prefix = model_filename.rstrip(".pkl")

    test_dataset = []
    with open(test_data_fn, 'r', encoding="utf-8") as test_data_f:
        for line in test_data_f:
            if all(c in model_runner.char2ix.keys() for c in line.strip()):
                test_dataset.append(line.strip())

    phonologyTool = PhonologyTool(phonology_features_filename)
    experimentCreator = ExperimentCreator(model_runner, test_dataset, phonologyTool)

    # front_harmony_dataset
    front_harmony_dataset_fn = model_filename_prefix + "_front_harmony_dataset.tsv"
    front_harmony_dataset = experimentCreator.make_dataset_pretty(experimentCreator.front_harmony_dataset())
    experimentCreator.save_dataset_to_tsv(front_harmony_dataset, front_harmony_dataset_fn)
    datasets.append(front_harmony_dataset_fn)

    return datasets


if torch.cuda.is_available():
    DEVICE_IX = 0
    torch.cuda.set_device(DEVICE_IX)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = ElmanRNN(len(char2index), HIDDEN_SIZE, len(char2index), device=device)

criterion = torch.nn.CrossEntropyLoss(
    ignore_index=char2index[WordData.WF_PAD_TOKEN])  # ignore_index=model.decoder_sos_token_index)
model = model.to(device)
optimizer = torch.optim.AdamW(
    params=model.parameters())  # torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
criterion = criterion.to(device)

MAX_EPOCHS = 500
STOP_DISTANCE = 10
STOP_NUMBER = 7

# liveloss = PlotLosses(groups={"loss": ["train", "test"]})
epoch_n = 0

task_fn2accuracy = {}
logs = []
test_losses = [float("inf")] * STOP_DISTANCE
for epoch in range(epoch_n, MAX_EPOCHS):
    epoch_losses = {}

    train_loss = train(model, train_loader, optimizer, criterion, char2index=char2index, device=device)
    test_loss = test(model, test_loader, criterion, char2index=char2index, device=device)
    test_losses.append(train_loss)

    # liveloss.update({"train": train_loss, "test": test_loss})

    print(f"lang: {LANG}", f"epoch: {epoch}", f"train_loss: {train_loss}", f"test_loss: {test_loss}", sep='\t')
    logs.append({"train": train_loss, "test": test_loss, "epoch": epoch})

    # if epoch % 1 == 0:
    #     liveloss.send()

    if epoch % 10 == 0:
        subprocess.call(["bash", "-c", "rm *{prediction,balanced.tsv}"])

    if epoch % 20 == 0:
        model_runner = ModelRunner(model, char2index, device)
        # model_runner.run_model_on_word("bare")
        datasets = build_datasets(model_runner, model_filename="turmodel_size_6_epoch_" + str(epoch))

        for task_fn in datasets:
            for hash in map(str, range(1)):
                with_agg_v_vals = (True, False)
                with_agg_v_vals = (False,)
                for with_agg_v in with_agg_v_vals:
                    probing_metainfo = dt_probe_dataset(task_fn, hash + 'TREE' + str(with_agg_v), with_agg=with_agg_v,
                                                        tree=True)
                    probing_metainfo["hash"] = hash
                    task_fn2accuracy[(task_fn, hash, with_agg_v, "TREE")] = probing_metainfo

                    probing_metainfo = dt_probe_dataset(task_fn, hash + 'LREG' + str(with_agg_v), with_agg=with_agg_v,
                                                        tree=False)
                    probing_metainfo["hash"] = hash
                    task_fn2accuracy[(task_fn, hash, with_agg_v, "LREG")] = probing_metainfo


res_f = open(f"{LANG}_{HIDDEN_SIZE}.log.pkl", 'w')
pickle.dump({"logs": logs, "taskfn2accuracy": task_fn2accuracy, "HIDDEN": HIDDEN_SIZE, "LANG":LANG}, res_f)
res_f.close()
