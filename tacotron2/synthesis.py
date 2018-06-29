#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import torch
from chinese2pinyin import ch2p
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from hparams import create_hparams


class TaGenerate(object):
    def __init__(self, tacotron_model_path, wavenet_model_path=None):
        self.hparams = create_hparams()
        self.taco_path = tacotron_model_path
        self.nv_wave_path = wavenet_model_path
        self.restore_tacotron()

    # TODO wavenet restore
    def restore_tacotron(self):
        checkpoint_path = self.taco_path
        self.model = load_model(self.hparams)
        try:
            self.model = self.model.module
        except:
            pass
        self.model.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load(checkpoint_path)['state_dict'].items()})
        _ = self.model.eval()

    def generate(self, text=None):
        text = ch2p(text)
        sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()
        mel_outputs, mel_outputs_postnet, _, alignments = self.model.inference(sequence)
        taco_stft = TacotronSTFT(
            self.hparams.filter_length, self.hparams.hop_length, self.hparams.win_length,
            sampling_rate=self.hparams.sampling_rate)
        mel_decompress = taco_stft.spectral_de_normalize(mel_outputs_postnet)
        mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
        spec_from_mel_scaling = 1000
        spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
        spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
        spec_from_mel = spec_from_mel * spec_from_mel_scaling
        waveform = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1]), taco_stft.stft_fn, 60)
        

