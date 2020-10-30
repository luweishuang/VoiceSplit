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
from utils.generic_utils import load_config_from_str, load_config, mix_wavfiles
from utils.demo_utils import save_spec, adjust_2_wavs, rm_wavs, rm_flacs
from utils.demo_utils_SNR import permute_SI_SNR, permutation_sdr


def get_embedding(encoder_model, ap, wave_file_path):
    preprocessed_wav = encoder_model.preprocess_wav(wave_file_path)
    file_embedding = encoder_model.embed_utterance(preprocessed_wav)
    return torch.from_numpy(file_embedding.reshape(-1))


def normalise_and_extract_features(encoder_model, ap, mixed_path, target_path, target_path2, emb_ref_path):
    mixed_path_norm = mixed_path.replace('.wav', '-norm.wav')
    target_path_norm = target_path.replace('.wav', '-norm.wav')
    target_path_norm2 = target_path2.replace('.wav', '-norm.wav')
    emb_ref_path_norm = emb_ref_path.replace('.wav', '-norm.wav')

    # normalise wavs
    os.system("ffmpeg-normalize %s -ar 16000 -o %s -f" % (mixed_path, mixed_path_norm))
    os.system("ffmpeg-normalize  %s -ar 16000 -o %s -f" % (target_path, target_path_norm))
    os.system("ffmpeg-normalize  %s -ar 16000 -o %s -f" % (target_path2, target_path_norm2))
    os.system("ffmpeg-normalize  %s -ar 16000 -o %s -f" % (emb_ref_path, emb_ref_path_norm))

    # load wavs
    target_wav = ap.load_wav(target_path_norm)
    target_wav2 = ap.load_wav(target_path_norm2)
    mixed_wav = ap.load_wav(mixed_path_norm)
    emb_wav = ap.load_wav(emb_ref_path_norm)

    # trim initial and end  wave file silence using librosa
    # target_wav, _ = librosa.effects.trim(target_wav, top_db=20)
    # mixed_wav, _ = librosa.effects.trim(mixed_wav, top_db=20)
    # emb_wav, _ = librosa.effects.trim(emb_wav, top_db=20)

    # normalise wavs
    norm_factor = np.max(np.abs(mixed_wav)) * 1.1
    mixed_wav = mixed_wav / norm_factor
    emb_wav = emb_wav / norm_factor
    target_wav = target_wav / norm_factor
    target_wav2 = target_wav2 / norm_factor

    # save embedding ref
    soundfile.write(emb_ref_path_norm, emb_wav, 16000)
    soundfile.write(mixed_path_norm, mixed_wav, 16000)
    soundfile.write(target_path_norm, target_wav, 16000)
    soundfile.write(target_path_norm2, target_wav2, 16000)

    embedding = get_embedding(encoder_model, ap, emb_ref_path_norm)
    mixed_spec, mixed_phase = ap.get_spec_from_audio(mixed_wav, return_phase=True)
    return embedding, mixed_spec, mixed_phase, target_wav, target_wav2, mixed_wav, emb_wav


def predict(encoder_model, ap, mixed_path, target_path, target_path2, emb_ref_path, outpath='predict.wav', save_img=False):
    embedding, mixed_spec, mixed_phase, target_wav, target_wav2, mixed_wav, emb_wav = normalise_and_extract_features(encoder_model, ap, mixed_path, target_path, target_path2, emb_ref_path)
    # use the model
    mixed_spec = torch.from_numpy(mixed_spec).float()

    # append 1 dimension on mixed, its need because the model spected batch
    mixed_spec = mixed_spec.unsqueeze(0)
    embedding = embedding.unsqueeze(0)

    if cuda:
        embedding = embedding.cuda()
        mixed_spec = mixed_spec.cuda()

    mask = model(mixed_spec, embedding)
    output = mixed_spec * mask

    # inverse spectogram to wav
    est_mag = output[0].cpu().detach().numpy()
    mixed_spec = mixed_spec[0].cpu().detach().numpy()
    # use phase from mixed wav for reconstruct the wave
    est_wav = ap.inv_spectrogram(est_mag, phase=mixed_phase)

    soundfile.write(outpath, est_wav, 16000)
    if save_img:
        img_path = outpath.replace('predict', 'images').replace(' ', '').replace('.wav', '-est.png')
        save_spec(img_path, est_mag)
        target_mag = ap.get_spec_from_audio(target_wav, return_phase=False)
        img_path = outpath.replace('predict', 'images').replace(' ', '').replace('.wav', '-target.png')
        save_spec(img_path, target_mag)
        img_path = outpath.replace('predict', 'images').replace(' ', '').replace('.wav', '-mixed.png')
        save_spec(img_path, mixed_spec)

    return est_wav, target_wav, target_wav2, mixed_wav, emb_wav


print("... Load GE2E encoder model ...")
# encoder.load_model(Path('encoder/saved_models/zh/aishell2_2.pt'))
encoder.load_model(Path('encoder/saved_models/pretrained_en.pt'))    # shell2数据集上，英文encoder的snr指标好于中文encoder模型

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

# dataset_name = 'datasets/aishell2'
# rm_wavs(Path(dataset_name), audio_len)  # do once
# exit()
# all_folders = [x for x in glob.glob(os.path.join(Path(dataset_name), 'audios/*'))]

dataset_name = 'datasets/libriSpeech/test-clean'     # libriSpeech 音频平均时长明显大于aishell2
# rm_flacs(Path(dataset_name), audio_len)  # do once
# exit()
all_folders = [x for x in glob.glob(os.path.join(Path(dataset_name), 'audios/*/*'))]

all_spk = [glob.glob(os.path.join(Path(spk), "**-norm.wav"), recursive=True) for spk in all_folders]
all_spk = [x for x in all_spk if len(x) >= 2]
spk1, spk2 = random.sample(all_spk, 2)
s1_dvec, s1_target = random.sample(spk1, 2)
s2 = random.choice(spk2)
print('s1_dvec: %s, s1_target: %s, s2: %s' % (s1_dvec, s1_target, s2))

output_dir = os.path.join(Path(dataset_name), 'output')
os.makedirs(output_dir, exist_ok=True)
mix_wav_path, s1_target_wav_path, s2_target_wav_path, s1_dvec_wav_path = mix_wavfiles(output_dir, sample_rate, audio_len, ap, form, 1, s1_dvec, s2, s1_target)

output_path = mix_wav_path.replace('mix', 'predict')
est_wav, target_wav, target_wav2, mixed_wav, emb_wav = predict(encoder, ap, mix_wav_path, s1_target_wav_path, s2_target_wav_path, s1_dvec_wav_path, outpath=output_path, save_img=False)
est_wav, mixed_wav = adjust_2_wavs(est_wav, mixed_wav)

# get wav for second voice, its need for SDR calculation
est_wav2 = mixed_wav - est_wav
est_wav2, mixed_wav = adjust_2_wavs(est_wav2, mixed_wav)

target_wav, mixed_wav = adjust_2_wavs(target_wav, mixed_wav)
# get target_wav for second voice, its recomended because google dont provide clean_utterance2 in your demo i need get in LibreSpeech Dataset, but i dont know if they normalised this file..
target_wav2 = mixed_wav - target_wav

# calculate snr and sdr before model
ests = [torch.from_numpy(mixed_wav), torch.from_numpy(mixed_wav)]  # the same voices is mixed_wav
egs = [torch.from_numpy(target_wav), torch.from_numpy(target_wav2)]
mix = torch.from_numpy(mixed_wav)
_snr, per = permute_SI_SNR(ests, egs, mix)
_sdr = permutation_sdr(ests, egs, mix, per)
print('=' * 20, "Before Model", '=' * 20)
print('SNRi: {:.5f}'.format(_snr))
print('SDRi: {:.5f}'.format(_sdr))
# calculate snr and sdr after model
ests = [torch.from_numpy(est_wav), torch.from_numpy(est_wav2)]
egs = [torch.from_numpy(target_wav), torch.from_numpy(target_wav2)]
mix = torch.from_numpy(mixed_wav)
_snr, per = permute_SI_SNR(ests, egs, mix)
_sdr = permutation_sdr(ests, egs, mix, per)
print('=' * 20, "after Model", '=' * 20)
print('SNRi: {:.5f}'.format(_snr))
print('SDRi: {:.5f}'.format(_sdr))

del target_wav, est_wav, mixed_wav