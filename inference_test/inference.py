import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import IPython.display as ipd

import sys
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
import audio

hparams = create_hparams()
hparams.sampling_rate = 22050

checkpoint_path = "tacotron2_statedict.pt"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.eval()

text = "Hi! what is your name?"
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()

# print(sequence)
mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
print(mel_outputs_postnet.size())
print(mel_outputs.size())

mel_outputs_postnet = mel_outputs_postnet.data.cpu().numpy()[0]
print(np.shape(mel_outputs_postnet))
# import audio

wav = audio.inv_mel_spectrogram(mel_outputs_postnet)
print(np.shape(wav))

audio.save_wav(wav, "tactron2.wav")
