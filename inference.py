import torch
import torch.nn as nn

import os
import numpy as np

from network import Tacotron2
from text import text_to_sequence
import hparams as hp
import audio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_path = "checkpoint_170100.pth.tar"
model = nn.DataParallel(Tacotron2(hp)).to(device)
model.load_state_dict(torch.load(os.path.join(
    hp.checkpoint_path, checkpoint_path))['model'])
model.eval()

with torch.no_grad():
    text = "What is your name?"
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    mel_outputs, mel_outputs_postnet, _, alignments = model.module.inference(
        sequence)

print(mel_outputs_postnet.size())
print(mel_outputs.size())

wav = audio.inv_mel_spectrogram(mel_outputs_postnet[0].cpu().numpy())
audio.save_wav(wav, "test_1.wav")

wav = audio.inv_mel_spectrogram(mel_outputs[0].cpu().numpy())
audio.save_wav(wav, "test_2.wav")
