# utils for plot spectrogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import os, glob
import sox
from pathlib import Path
import imageio


def rm_wavs111(wav_dir, audio_len_thread):
    all_folders = [x for x in glob.glob(os.path.join(wav_dir, 'audios/*'))]
    all_spk = [glob.glob(os.path.join(Path(spk), "**.wav"), recursive=True) for spk in all_folders]
    for cur_spk in all_spk:
        for cur_wav_path in cur_spk:
            cur_audio_len = sox.file_info.duration(cur_wav_path)
            if cur_audio_len < audio_len_thread+2:
                os.system("rm %s" % cur_wav_path)
    return


def rm_flacs(wav_dir, audio_len_thread):
    all_folders = [x for x in glob.glob(os.path.join(wav_dir, 'audios/*/*'))]
    all_spk = [glob.glob(os.path.join(Path(spk), "**-norm.wav"), recursive=True) for spk in all_folders]
    for cur_spk in all_spk:
        for cur_norm_wav_path in cur_spk:
            cur_wav_path = cur_norm_wav_path.replace("-norm.wav", ".flac")
            cur_audio_len = sox.file_info.duration(cur_norm_wav_path)
            if cur_audio_len < audio_len_thread+2:
                os.system("rm %s" % cur_wav_path)
                os.system("rm %s" % cur_norm_wav_path)
    return


def rm_wavs(wav_dir, audio_len_thread, ext=".wav"):
    all_folders = [x for x in glob.glob(os.path.join(wav_dir, 'audios'))]
    all_spk = [glob.glob(os.path.join(Path(spk), "**-norm.wav"), recursive=True) for spk in all_folders]
    for cur_spk in all_spk:
        for cur_norm_wav_path in cur_spk:
            cur_wav_path = cur_norm_wav_path.replace("-norm.wav", ".wav")
            cur_audio_len = sox.file_info.duration(cur_wav_path)
            if cur_audio_len < audio_len_thread+2:
                os.system("rm %s" % cur_wav_path)
                os.system("rm %s" % cur_norm_wav_path)
    return


def adjust_2_wavs(est_wav, mixed_wav):
    len_est = len(est_wav)
    len_mixed = len(mixed_wav)
    if len_est > len_mixed:
        # mixed need is biggest
        est_wav = est_wav[:len_mixed]
    else:
        # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
        est_wav = np.pad(est_wav, (0, len_mixed - len_est), 'constant', constant_values=(0, 0))
    return est_wav, mixed_wav


def fig2np(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect='auto', origin='lower',
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel('Frames')
    plt.ylabel('Channels')
    plt.tight_layout()

    fig.canvas.draw()
    data = fig2np(fig)
    plt.close()
    return data


def save_spec(path, spec):
  data = plot_spectrogram_to_numpy(spec)
  imageio.imwrite(path, data)