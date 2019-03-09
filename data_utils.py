import torch
from torch.utils.data import Dataset, DataLoader
from text import text_to_sequence
import hparams

import numpy as np
import os


device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')


class Tacotron2DataLoader(Dataset):
    """LJSpeech"""

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.textPath = os.path.join(self.dataset_path, "train.txt")
        self.text = process_text(self.textPath)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        # print("ljspeech-mel-%05d.npy" % (index + 1))
        mel_name = os.path.join(
            self.dataset_path, "ljspeech-mel-%05d.npy" % (index + 1))
        character = self.text[index]
        character = text_to_sequence(character, hparams.text_cleaners)
        character = np.array(character)
        mel_np = np.load(mel_name)

        return {"text": character, "mel": mel_np}


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8")as f:
        inx = 0
        txt = []
        for line in f.readlines():
            cnt = 0
            for index, ele in enumerate(line):
                if ele == '|':
                    cnt = cnt + 1
                    if cnt == 3:
                        inx = index
                        end = len(line)
                        txt.append(line[inx + 1:end-1])
                        break
        return txt


def collate_fn(batch):

    texts = [d['text'] for d in batch]
    mels = [d['mel'] for d in batch]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, np.shape(text)[0])

    length_mel = np.array([])
    for mel in mels:
        length_mel = np.append(length_mel, np.shape(mel)[0])

    texts = pad_seq_text(texts)
    mels = pad_seq_spec(mels)

    index = np.argsort(- length_text)
    new_texts = np.stack([texts[i] for i in index])
    length_text = np.stack([length_text[i]
                            for i in np.argsort(- length_text)]).astype(np.int32)

    index = np.argsort(- length_mel)
    new_mels = np.stack([np.transpose(mels[i])for i in index])
    length_mel = np.stack([length_mel[i]
                           for i in np.argsort(- length_mel)]).astype(np.int32)

    total_len = np.shape(new_mels)[2]
    gate_padded = np.stack([gen_gate(total_len, length_mel[i])
                            for i in range(np.shape(new_mels)[0])])

    # if torch.cuda.is_available():
    #     new_texts = torch.from_numpy(new_texts).type(
    #         torch.cuda.LongTensor).to(device)
    # else:
    #     new_texts = torch.from_numpy(new_texts).type(
    #         torch.LongTensor).to(device)
    # new_mels = torch.from_numpy(new_mels).to(device)

    # gate_padded = torch.from_numpy(gate_padded).to(device)

    # length_text = torch.from_numpy(length_text).to(device)
    # length_mel = torch.from_numpy(length_mel).to(device)

    return new_texts, length_text, new_mels, gate_padded, length_mel


def gen_gate(total_len, target_len):
    # print(target_len)
    # print(total_len)
    out = np.array([0 for i in range(total_len)])
    for i in range(target_len-1, total_len):
        out[i] = 1

    return out


def pad_seq_text(inputs):
    def pad_data(x, length):
        pad = 0
        return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=pad)

    max_len = max((len(x)for x in inputs))
    return np.stack([pad_data(x, max_len)for x in inputs])


def pad_seq_spec(inputs):
    def pad(x, max_len):
        # print(type(x))
        if np.shape(x)[0] > max_len:
            # print("ERROR!")
            raise ValueError("not max_len")
        s = np.shape(x)[1]
        # print(s)
        x = np.pad(x, (0, max_len - np.shape(x)
                       [0]), mode='constant', constant_values=0)
        return x[:, :s]

    max_len = max(np.shape(x)[0] for x in inputs)
    return np.stack([pad(x, max_len)for x in inputs])


if __name__ == "__main__":

    dataset = Tacotron2DataLoader("dataset_LJSpeech")
    train_loader = DataLoader(dataset, num_workers=1, shuffle=True,
                              batch_size=hparams.batch_size, drop_last=True, collate_fn=collate_fn)

    print(len(train_loader))
    for i, batch in enumerate(train_loader):

        if i == 1:
            break

        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch

        print("text", np.shape(text_padded))
        print("input length", np.shape(input_lengths))
        print(input_lengths)
        print("mel", np.shape(mel_padded))
        print("gate", np.shape(gate_padded))
        print("mel length", output_lengths)
        print(gate_padded)

        cnt = 0
        for i in gate_padded[0]:
            if i == 1:
                cnt = cnt + 1
        print(cnt)

        cnt = 0
        for i in gate_padded[1]:
            if i == 1:
                cnt = cnt + 1
        print(cnt)
