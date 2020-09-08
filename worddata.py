from tqdm.autonotebook import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

class WordData(Dataset):
    WF_PAD_TOKEN = "CPAD"
    WF_EOS_TOKEN = "EOS"

    def __init__(self, src_data, tgt_data, char2index,
                 sequence_length=20,
                 #    wf_pad_token='CPAD',
                 #    eos_token='EOS',
                 verbose=True):

        super().__init__()

        self.src_data = []
        self.tgt_data = []
        self.char2index = char2index
        self.index2char = {v: k for k, v in char2index.items()}

        self.sequence_length = sequence_length or max(map(len, src_data)) + 1
        self.load(zip(src_data, tgt_data))

    def load(self, data, verbose=True):

        data_iterator = tqdm(data, desc='Loading data', disable=not verbose)

        for src, tgt in data_iterator:
            src = list(src)
            tgt = list(tgt)

            if len(src) + 1 > self.sequence_length or len(tgt) > self.sequence_length:
                continue

            src += [self.WF_EOS_TOKEN] + [self.WF_PAD_TOKEN] * (self.sequence_length - len(src) - 1)
            tgt += [self.WF_EOS_TOKEN] + [self.WF_PAD_TOKEN] * (self.sequence_length - len(tgt) - 1)

            indexed_src = self.indexing_wf(src)
            indexed_tgt = self.indexing_wf(tgt)
            assert len(indexed_src) == len(indexed_tgt) == self.sequence_length
            self.src_data.append(indexed_src)
            self.tgt_data.append(indexed_tgt)

    def indexing_wf(self, wf):
        return [self.char2index[c] for c in wf if c in self.char2index]

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):

        src_item = self.src_data[idx]
        src_item = torch.Tensor(src_item).long()

        tgt_item = self.tgt_data[idx]
        tgt_item = torch.Tensor(tgt_item).long()

        assert len(src_item) == len(tgt_item) == self.sequence_length
        # print(src_item.shape)
        # print(tgt_item.shape)

        return src_item, tgt_item