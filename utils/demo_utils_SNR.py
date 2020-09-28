# utils for calculate SNR and SDR
# this code is adpated from https://github.com/JusperLee/Calculate-SNR-SDR/
import torch
from mir_eval.separation import bss_eval_sources
from itertools import permutations


def SI_SNR(_s, s, mix, zero_mean=True):
    '''
         Calculate the SNR indicator between the two audios.
         The larger the value, the better the separation.
         input:
               _s: Generated audio
               s:  Ground Truth audio
         output:
               SNR value
    '''
    length = _s.shape[0]
    _s = _s[:length]
    s =s[:length]
    mix = mix[:length]
    if zero_mean:
        _s = _s - torch.mean(_s)
        s = s - torch.mean(s)
        mix = mix - torch.mean(mix)
    s_target = sum(torch.mul(_s, s))*s/(torch.pow(torch.norm(s, p=2), 2)+1e-8)
    e_noise = _s - s_target
    # mix ---------------------------
    mix_target = sum(torch.mul(mix, s))*s/(torch.pow(torch.norm(s, p=2), 2)+1e-8)
    mix_noise = mix - mix_target
    return 20*torch.log10(torch.norm(s_target, p=2)/(torch.norm(e_noise, p=2)+1e-8)) - 20*torch.log10(torch.norm(mix_target, p=2)/(torch.norm(mix_noise, p=2)+1e-8))


def permute_SI_SNR(_s_lists, s_lists, mix):
    '''
        Calculate all possible SNRs according to
        the permutation combination and
        then find the maximum value.
        input:
               _s_lists: Generated audio list
               s_lists: Ground truth audio list
        output:
               max of SI-SNR
    '''
    length = len(_s_lists)
    results = []
    per = []
    for p in permutations(range(length)):
        s_list = [s_lists[n] for n in p]
        result = sum([SI_SNR(_s, s, mix, zero_mean=True) for _s, s in zip(_s_lists, s_list)])/length
        results.append(result)
        per.append(p)
    return max(results), per[results.index(max(results))]


def SDR(est, egs, mix):
    '''
        calculate SDR
        est: Network generated audio
        egs: Ground Truth
    '''
    length = est.numpy().shape[0]
    sdr, _, _, _ = bss_eval_sources(egs.numpy()[:length], est.numpy()[:length])
    mix_sdr, _, _, _ = bss_eval_sources(egs.numpy()[:length], mix.numpy()[:length])
    return float(sdr-mix_sdr)


def permutation_sdr(est_list, egs_list, mix, per):
    n = len(est_list)
    result = sum([SDR(est_list[a], egs_list[b], mix) for a, b in enumerate(per)])/n
    return result
