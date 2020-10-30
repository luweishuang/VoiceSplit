# -*- coding: utf-8 -*-
import torch
import soundfile
import random
import os
import numpy as np
import glob

from encoder import inference as encoder
from pathlib import Path

# Imports from VoiceSplit model
from utils.audio_processor import WrapperAudioProcessor as AudioProcessor
from models.voicefilter.model import VoiceFilter
from models.voicesplit.model import VoiceSplit
from utils.generic_utils import load_config_from_str


def get_embedding(encoder_model, wave_file_path):
    preprocessed_wav = encoder_model.preprocess_wav(wave_file_path)
    file_embedding = encoder_model.embed_utterance(preprocessed_wav)
    return torch.from_numpy(file_embedding.reshape(-1))


def normalise_and_extract_features_one_person(encoder_model, ap, mixed_path, target_path, emb_ref_path, output_dir):
    mixed_path_norm = os.path.join(output_dir, os.path.basename(mixed_path))
    target_path_norm = os.path.join(output_dir, os.path.basename(target_path))
    emb_ref_path_norm = os.path.join(output_dir, os.path.basename(emb_ref_path))

    # load wavs
    target_wav = ap.load_wav(target_path)
    mixed_wav = ap.load_wav(mixed_path)
    emb_wav = ap.load_wav(emb_ref_path)

    # normalise wavs
    norm_factor = np.max(np.abs(mixed_wav)) * 1.1
    mixed_wav = mixed_wav / norm_factor
    emb_wav = emb_wav / norm_factor
    target_wav = target_wav / norm_factor

    # save embedding ref
    soundfile.write(emb_ref_path_norm, emb_wav, 16000)
    soundfile.write(mixed_path_norm, mixed_wav, 16000)
    soundfile.write(target_path_norm, target_wav, 16000)

    embedding = get_embedding(encoder_model, emb_ref_path_norm)
    mixed_spec, mixed_phase = ap.get_spec_from_audio(mixed_wav, return_phase=True)
    os.system("rm %s" % emb_ref_path_norm)
    return embedding, mixed_spec, mixed_phase, target_wav, mixed_wav, emb_wav


def predict_one_person(encoder_model, ap, mixed_path, target_path, emb_ref_path, output_dir):
    embedding, mixed_spec, mixed_phase, target_wav, mixed_wav, emb_wav = normalise_and_extract_features_one_person(encoder_model, ap, mixed_path, target_path, emb_ref_path, output_dir)
    # use the model
    mixed_spec = torch.from_numpy(mixed_spec).float()

    # append 1 dimension on mixed, its need because the model spected batch
    mixed_spec = mixed_spec.unsqueeze(0)
    embedding = embedding.unsqueeze(0)

    mask = model(mixed_spec, embedding)
    output = mixed_spec * mask

    # inverse spectogram to wav
    est_mag = output[0].cpu().detach().numpy()
    # use phase from mixed wav for reconstruct the wave
    est_wav = ap.inv_spectrogram(est_mag, phase=mixed_phase)

    outpath = os.path.join(output_dir, os.path.basename(mixed_path).replace(".wav", "_predict.wav"))
    soundfile.write(outpath, est_wav, 16000)
    return est_wav, target_wav, mixed_wav, emb_wav


print("... Load GE2E encoder model ...")
encoder.load_model(Path('encoder/saved_models/pretrained_en.pt'))

checkpoint_path = 'models/demo5.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model_c = load_config_from_str(checkpoint['config_str'])

ap = AudioProcessor(model_c.audio)  # create AudioProcessor for model
model_name = model_c.model_name
cuda = False
if model_name == 'voicefilter':
    print('inicializado com voicefilter')
    model = VoiceFilter(model_c)
elif model_name == 'voicesplit':
    model = VoiceSplit(model_c)
else:
    raise Exception(" The model '" + model_name + "' is not suported")
if model_c.train_config['optimizer'] == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=model_c.train_config['learning_rate'])
else:
    raise Exception("The %s  not is a optimizer supported" % model_c.train['optimizer'])
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
step = checkpoint['step']
print("load model form Step:", step)
if cuda:
    model = model.cuda()

sample_rate = model_c.audio[model_c.audio['backend']]['sample_rate']
audio_len = model_c.audio['audio_len']
form = model_c.dataset['format']

dataset_name = 'datasets/aishell2'
all_folders = [x for x in glob.glob(os.path.join(Path(dataset_name), 'audios/*'))]

all_spk = [glob.glob(os.path.join(Path(spk), "**-norm.wav"), recursive=True) for spk in all_folders]
all_spk = [x for x in all_spk if len(x) >= 2]
for spk1 in all_spk:
    # spk1 = random.sample(all_spk, 1)
    s1_dvec, s1_target = random.sample(spk1, 2)
    print('s1_dvec: %s, s1_target: %s' % (s1_dvec, s1_target))

    output_dir = os.path.join(Path(dataset_name), 'output')
    os.makedirs(output_dir, exist_ok=True)
    mix_wav_path = s1_target
    s1_target_wav_path = s1_target
    s1_dvec_wav_path = s1_dvec
    est_wav, target_wav, mixed_wav, emb_wav = predict_one_person(encoder, ap, mix_wav_path, s1_target_wav_path, s1_dvec_wav_path, output_dir)