# -*- coding: utf-8 -*-
import torch
import soundfile
import librosa
import os
import numpy as np
import pandas as pd

# Imports from GE2E
from encoder import inference as encoder
from pathlib import Path

# Imports from VoiceSplit model
from utils.audio_processor import WrapperAudioProcessor as AudioProcessor
from models.voicefilter.model import VoiceFilter
from models.voicesplit.model import VoiceSplit
from utils.generic_utils import load_config_from_str
from utils.demo_utils import save_spec
from utils.demo_utils_SNR import permute_SI_SNR, permutation_sdr
from IPython.display import Audio, display


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


def voiceFilter_2_speaker():
    # create output path
    os.makedirs('datasets/LibriSpeech/audios_demo/2_speakers/predict/', exist_ok=True)
    os.makedirs('datasets/LibriSpeech/audios_demo/2_speakers/images/', exist_ok=True)

    test_csv = pd.read_csv('datasets/LibriSpeech/test_demo.csv', sep=',').values

    sdrs_before = []
    sdrs_after = []
    snrs_before = []
    snrs_after = []
    for noise_utterance, emb_utterance, clean_utterance, clean_utterance2 in test_csv:
        noise_utterance = noise_utterance.replace(' ', '')
        emb_utterance = emb_utterance.replace(' ', '')
        clean_utterance = clean_utterance.replace(' ', '')
        clean_utterance2 = clean_utterance2.replace(' ', '')
        output_path = noise_utterance.replace('noisy', 'predict').replace(' ', '')
        est_wav, target_wav, target_wav2, mixed_wav, emb_wav = predict(encoder, ap, noise_utterance, clean_utterance, clean_utterance2, emb_utterance, outpath=output_path, save_img=True)

        len_est = len(est_wav)
        len_mixed = len(mixed_wav)
        if len_est > len_mixed:
            # mixed need is biggest
            est_wav = est_wav[:len_mixed]
        else:
            # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
            est_wav = np.pad(est_wav, (0, len(mixed_wav) - len(est_wav)), 'constant', constant_values=(0, 0))

        # get wav for second voice, its need for SDR calculation
        est_wav2 = mixed_wav - est_wav

        len_est = len(est_wav2)
        len_mixed = len(mixed_wav)
        if len_est > len_mixed:
            # mixed need is biggest
            est_wav2 = est_wav2[:len_mixed]
        else:
            # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
            est_wav2 = np.pad(est_wav2, (0, len(mixed_wav) - len(est_wav2)), 'constant', constant_values=(0, 0))

        len_est = len(target_wav)
        len_mixed = len(mixed_wav)
        if len_est > len_mixed:
            # mixed need is biggest
            target_wav = target_wav[:len_mixed]
        else:
            # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
            target_wav = np.pad(target_wav, (0, len(mixed_wav) - len(target_wav)), 'constant', constant_values=(0, 0))

        # get target_wav for second voice, its recomended because google dont provide clean_utterance2 in your demo i need get in LibreSpeech Dataset, but i dont know if they normalised this file..
        target_wav2 = mixed_wav - target_wav

        # calculate snr and sdr before model
        ests = [torch.from_numpy(mixed_wav), torch.from_numpy(mixed_wav)]  # the same voices is mixed_wav
        egs = [torch.from_numpy(target_wav), torch.from_numpy(target_wav2)]
        mix = torch.from_numpy(mixed_wav)
        _snr, per = permute_SI_SNR(ests, egs, mix)
        _sdr = permutation_sdr(ests, egs, mix, per)
        snrs_before.append(_snr)
        sdrs_before.append(_sdr)

        # calculate snr and sdr after model
        ests = [torch.from_numpy(est_wav), torch.from_numpy(est_wav2)]
        egs = [torch.from_numpy(target_wav), torch.from_numpy(target_wav2)]
        mix = torch.from_numpy(mixed_wav)
        _snr, per = permute_SI_SNR(ests, egs, mix)
        _sdr = permutation_sdr(ests, egs, mix, per)
        snrs_after.append(_snr)
        sdrs_after.append(_sdr)

        # show in notebook results
        print('-' * 100)
        print('-' * 30, os.path.basename(noise_utterance), '-' * 30)
        print("Input/Noise Audio")
        display(Audio(mixed_wav, rate=16000))
        print('Predicted Audio')
        display(Audio(est_wav, rate=16000))
        print('Target Audio')
        display(Audio(target_wav, rate=16000))
        print('Predicted2 Audio')
        display(Audio(est_wav2, rate=16000))
        print('Target2 Audio')
        display(Audio(target_wav2, rate=16000))
        print('-' * 100)
        del target_wav, est_wav, mixed_wav

    print('=' * 20, "Before Model", '=' * 20)
    print('\nAverage SNRi: {:.5f}'.format(np.array(snrs_before).mean()))
    print('Average SDRi: {:.5f}'.format(np.array(sdrs_before).mean()))

    print('=' * 20, "After Model", '=' * 20)
    print('\nAverage SNRi: {:.5f}'.format(np.array(snrs_after).mean()))
    print('Average SDRi: {:.5f}'.format(np.array(sdrs_after).mean()))


def voiceFilter_single_speaker():
    os.makedirs('datasets/LibriSpeech/audios_demo/single_speaker/predict/', exist_ok=True)
    os.makedirs('datasets/LibriSpeech/audios_demo/single_speaker/images/', exist_ok=True)
    test_csv = pd.read_csv('datasets/LibriSpeech/test_demo.csv', sep=',').values

    sdrs_before = []
    sdrs_after = []
    snrs_before = []
    snrs_after = []
    for noise_utterance, emb_utterance, clean_utterance, clean_utterance2 in test_csv:
        emb_utterance = os.path.join('VoiceSplit', emb_utterance).replace(' ', '')
        clean_utterance = os.path.join('VoiceSplit', clean_utterance).replace(' ', '')
        clean_utterance2 = os.path.join('VoiceSplit', clean_utterance2).replace(' ', '')
        output_path = clean_utterance.replace('/clean/', '/single_speaker/predict/').replace(' ', '')

        #  input = clean uterrance
        est_wav, target_wav, target_wav2, mixed_wav, emb_wav = predict(encoder, ap, clean_utterance, clean_utterance, clean_utterance2, emb_utterance, outpath=output_path, save_img=True)

        len_est = len(est_wav)
        len_mixed = len(mixed_wav)
        if len_est > len_mixed:
            # mixed need is biggest
            est_wav = est_wav[:len_mixed]
        else:
            # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
            est_wav = np.pad(est_wav, (0, len(mixed_wav) - len(est_wav)), 'constant', constant_values=(0, 0))

        # get wav for second voice, its need for SDR calculation
        est_wav2 = mixed_wav - est_wav

        len_est = len(est_wav2)
        len_mixed = len(mixed_wav)
        if len_est > len_mixed:
            # mixed need is biggest
            est_wav2 = est_wav2[:len_mixed]
        else:
            # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
            est_wav2 = np.pad(est_wav2, (0, len(mixed_wav) - len(est_wav2)), 'constant', constant_values=(0, 0))

        len_est = len(target_wav)
        len_mixed = len(mixed_wav)
        if len_est > len_mixed:
            # mixed need is biggest
            target_wav = target_wav[:len_mixed]
        else:
            # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
            target_wav = np.pad(target_wav, (0, len(mixed_wav) - len(target_wav)), 'constant', constant_values=(0, 0))

        # show in notebook results
        print('-' * 100)
        print('-' * 30, os.path.basename(noise_utterance), '-' * 30)
        print("Input/Clean Audio")
        display(Audio(mixed_wav, rate=16000))
        print('Predicted Audio')
        display(Audio(est_wav, rate=16000))
        print('-' * 100)
        del target_wav, est_wav, mixed_wav


# SDR from google paper for this instances
def SDR_google():
    test_csv = pd.read_csv('VoiceSplit/datasets/LibriSpeech/test_demo.csv', sep=',').values
    sdrs_before = []
    sdrs_after = []
    snrs_after = []
    snrs_before = []
    for noise_utterance, emb_utterance, clean_utterance, clean_utterance2 in test_csv:
        noise_utterance = os.path.join('VoiceSplit', noise_utterance).replace(' ', '')
        emb_utterance = os.path.join('VoiceSplit', emb_utterance).replace(' ', '')
        clean_utterance = os.path.join('VoiceSplit', clean_utterance).replace(' ', '')
        clean_utterance2 = os.path.join('VoiceSplit', clean_utterance2).replace(' ', '')
        est_utterance = noise_utterance.replace('noisy', 'enhanced').replace(' ', '')

        target_wav, _ = librosa.load(clean_utterance, sr=16000)
        target_wav2, _ = librosa.load(clean_utterance2, sr=16000)
        est_wav, _ = librosa.load(est_utterance, sr=16000)
        mixed_wav, _ = librosa.load(noise_utterance, sr=16000)

        len_est = len(est_wav)
        len_mixed = len(mixed_wav)
        if len_est > len_mixed:
            # mixed need is biggest
            est_wav = est_wav[:len_mixed]
        else:
            # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
            est_wav = np.pad(est_wav, (0, len(mixed_wav) - len(est_wav)), 'constant', constant_values=(0, 0))

        # get wav for second voice, its need for SDR calculation
        est_wav2 = mixed_wav - est_wav

        len_est = len(est_wav2)
        len_mixed = len(mixed_wav)
        if len_est > len_mixed:
            # mixed need is biggest
            est_wav2 = est_wav2[:len_mixed]
        else:
            # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
            est_wav2 = np.pad(est_wav2, (0, len(mixed_wav) - len(est_wav2)), 'constant', constant_values=(0, 0))

        len_est = len(target_wav)
        len_mixed = len(mixed_wav)
        if len_est > len_mixed:
            # mixed need is biggest
            target_wav = target_wav[:len_mixed]
        else:
            # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
            target_wav = np.pad(target_wav, (0, len(mixed_wav) - len(target_wav)), 'constant', constant_values=(0, 0))

        # get target_wav for second voice, its recomended because google dont provide clean_utterance2 in your demo i need get in LibreSpeech Dataset, but i dont know if they normalised this file..
        target_wav2 = mixed_wav - target_wav

        # calculate snr and sdr before model
        ests = [torch.from_numpy(mixed_wav), torch.from_numpy(mixed_wav)]  # the same voices is mixed_wav
        egs = [torch.from_numpy(target_wav), torch.from_numpy(target_wav2)]
        mix = torch.from_numpy(mixed_wav)
        _snr, per = permute_SI_SNR(ests, egs, mix)
        _sdr = permutation_sdr(ests, egs, mix, per)
        snrs_before.append(_snr)
        sdrs_before.append(_sdr)

        # calculate snr and sdr after model
        ests = [torch.from_numpy(est_wav), torch.from_numpy(est_wav2)]
        egs = [torch.from_numpy(target_wav), torch.from_numpy(target_wav2)]
        mix = torch.from_numpy(mixed_wav)
        _snr, per = permute_SI_SNR(ests, egs, mix)
        _sdr = permutation_sdr(ests, egs, mix, per)
        snrs_after.append(_snr)
        sdrs_after.append(_sdr)

        # show in notebook results
        print('-' * 100)
        print('-' * 30, os.path.basename(noise_utterance), '-' * 30)
        print("Input/Noise Audio")
        display(Audio(mixed_wav, rate=16000))
        print('Predicted Audio')
        display(Audio(est_wav, rate=16000))
        print('Target Audio')
        display(Audio(target_wav, rate=16000))
        print('Predicted2 Audio')
        display(Audio(est_wav2, rate=16000))
        print('Target2 Audio')
        display(Audio(target_wav2, rate=16000))
        print('-' * 100)
        del target_wav, est_wav, mixed_wav

    print('=' * 20, "Before Model", '=' * 20)
    print('\nAverage SNRi: {:.5f}'.format(np.array(snrs_before).mean()))
    print('Average SDRi: {:.5f}'.format(np.array(sdrs_before).mean()))

    print('=' * 20, "After Model", '=' * 20)
    print('\nAverage SNRi: {:.5f}'.format(np.array(snrs_after).mean()))
    print('Average SDRi: {:.5f}'.format(np.array(sdrs_after).mean()))


# Load GE2E model
print("Preparing the encoder, the synthesizer and the vocoder...")
encoder.load_model(Path('encoder/saved_models/zh/aishell2_2.pt'))
print("Testing your configuration with small inputs.")

checkpoint_path = 'models/demo5.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
c = load_config_from_str(checkpoint['config_str'])

ap = AudioProcessor(c.audio)  # create AudioProcessor for model
model_name = c.model_name
cuda = False
if model_name == 'voicefilter':
    print('inicializado com voicefilter')
    model = VoiceFilter(c)
elif model_name == 'voicesplit':
    model = VoiceSplit(c)
else:
    raise Exception(" The model '" + model_name + "' is not suported")
if c.train_config['optimizer'] == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=c.train_config['learning_rate'])
else:
    raise Exception("The %s  not is a optimizer supported" % c.train['optimizer'])
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
step = checkpoint['step']
print("load model form Step:", step)
if cuda:
    model = model.cuda()

# voiceFilter_2_speaker()
# voiceFilter_single_speaker()
# SDR_google()


# create output path
os.makedirs('datasets/LibriSpeech/audios_demo/2_speakers/predict/', exist_ok=True)
os.makedirs('datasets/LibriSpeech/audios_demo/2_speakers/images/', exist_ok=True)

test_csv = pd.read_csv('datasets/LibriSpeech/test_demo.csv', sep=',').values

sdrs_before = []
sdrs_after = []
snrs_before = []
snrs_after = []
for noise_utterance, emb_utterance, clean_utterance, clean_utterance2 in test_csv:
    noise_utterance = noise_utterance.replace(' ', '')
    emb_utterance = emb_utterance.replace(' ', '')
    clean_utterance = clean_utterance.replace(' ', '')
    clean_utterance2 = clean_utterance2.replace(' ', '')
    output_path = noise_utterance.replace('noisy', 'predict').replace(' ', '')
    est_wav, target_wav, target_wav2, mixed_wav, emb_wav = predict(encoder, ap, noise_utterance, clean_utterance, clean_utterance2, emb_utterance, outpath=output_path, save_img=True)

    len_est = len(est_wav)
    len_mixed = len(mixed_wav)
    if len_est > len_mixed:
        # mixed need is biggest
        est_wav = est_wav[:len_mixed]
    else:
        # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
        est_wav = np.pad(est_wav, (0, len(mixed_wav) - len(est_wav)), 'constant', constant_values=(0, 0))

    # get wav for second voice, its need for SDR calculation
    est_wav2 = mixed_wav - est_wav

    len_est = len(est_wav2)
    len_mixed = len(mixed_wav)
    if len_est > len_mixed:
        # mixed need is biggest
        est_wav2 = est_wav2[:len_mixed]
    else:
        # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
        est_wav2 = np.pad(est_wav2, (0, len(mixed_wav) - len(est_wav2)), 'constant', constant_values=(0, 0))

    len_est = len(target_wav)
    len_mixed = len(mixed_wav)
    if len_est > len_mixed:
        # mixed need is biggest
        target_wav = target_wav[:len_mixed]
    else:
        # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
        target_wav = np.pad(target_wav, (0, len(mixed_wav) - len(target_wav)), 'constant', constant_values=(0, 0))

    # get target_wav for second voice, its recomended because google dont provide clean_utterance2 in your demo i need get in LibreSpeech Dataset, but i dont know if they normalised this file..
    target_wav2 = mixed_wav - target_wav

    # calculate snr and sdr before model
    ests = [torch.from_numpy(mixed_wav), torch.from_numpy(mixed_wav)]  # the same voices is mixed_wav
    egs = [torch.from_numpy(target_wav), torch.from_numpy(target_wav2)]
    mix = torch.from_numpy(mixed_wav)
    _snr, per = permute_SI_SNR(ests, egs, mix)
    _sdr = permutation_sdr(ests, egs, mix, per)
    snrs_before.append(_snr)
    sdrs_before.append(_sdr)

    # calculate snr and sdr after model
    ests = [torch.from_numpy(est_wav), torch.from_numpy(est_wav2)]
    egs = [torch.from_numpy(target_wav), torch.from_numpy(target_wav2)]
    mix = torch.from_numpy(mixed_wav)
    _snr, per = permute_SI_SNR(ests, egs, mix)
    _sdr = permutation_sdr(ests, egs, mix, per)
    snrs_after.append(_snr)
    sdrs_after.append(_sdr)

    # show in notebook results
    print('-' * 100)
    print('-' * 30, os.path.basename(noise_utterance), '-' * 30)
    print("Input/Noise Audio")
    display(Audio(mixed_wav, rate=16000))
    print('Predicted Audio')
    display(Audio(est_wav, rate=16000))
    print('Target Audio')
    display(Audio(target_wav, rate=16000))
    print('Predicted2 Audio')
    display(Audio(est_wav2, rate=16000))
    print('Target2 Audio')
    display(Audio(target_wav2, rate=16000))
    print('-' * 100)
    del target_wav, est_wav, mixed_wav

print('=' * 20, "Before Model", '=' * 20)
print('\nAverage SNRi: {:.5f}'.format(np.array(snrs_before).mean()))
print('Average SDRi: {:.5f}'.format(np.array(sdrs_before).mean()))

print('=' * 20, "After Model", '=' * 20)
print('\nAverage SNRi: {:.5f}'.format(np.array(snrs_after).mean()))
print('Average SDRi: {:.5f}'.format(np.array(sdrs_after).mean()))