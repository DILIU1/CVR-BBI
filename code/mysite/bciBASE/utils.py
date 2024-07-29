# include some simple functions



def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
    """
    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE's PSD functions or channel/trial subsets.
    noise_n_neighbor_freqs : int
        Number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighbor_freqs : int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Array containing SNR for all epochs, channels, frequency bins.
        NaN for frequencies on the edges, that do not have enough neighbors on
        one side to calculate SNR.
    """
    averaging_kernel = np.concatenate((
        np.ones(noise_n_neighbor_freqs),
        np.zeros(2 * noise_skip_neighbor_freqs + 1),
        np.ones(noise_n_neighbor_freqs)))
    averaging_kernel /= averaging_kernel.sum()
    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode='valid'),
        axis=-1, arr=psd
    )
    # The mean is not defined on the edges so we will pad it with nas. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(
        mean_noise, pad_width=pad_width, constant_values=np.nan
    )
    return psd / mean_noise



import numpy as np
def max_freq(pfs,ps,return_f='max'):
    # ps[ij] > 0
    ps[np.isnan(ps)] = 0

    # return_f: 'max', 'center'
    if return_f == 'max':
        max_freqs = []
        max_freqs_values = []
        psds_len = ps.shape[0]
        for i in range(psds_len):
            # former channels will be first chosen by ssvep_judge then
            psds_index = np.argsort(ps[i,:])
            psds_freqs = pfs[psds_index].round()
            psds = ps[i,:][psds_index]
            max_freqs.append(psds_freqs[-1])
            max_freqs_values.append(psds[-1])
        # note to return only freqs
        return max_freqs, max_freqs_values
    
    elif return_f == 'center':
        # note to append center_mask function
        center_freqs = []
        psds_len = ps.shape[0]
        for ifreq in range(psds_len):
            sump = 0
            sumpf = 0
            for pf,p in zip(pfs,ps[ifreq,:]):
                if not np.isnan(p):
                    sump += p
                    sumpf += pf*p
            center_freqs.append(sumpf/sump)
        return center_freqs
    
    else:
        raise

from collections import Counter
def ssvep_judge(max_freqs,ssvep_freqs):
    try: 
        # NOTE that if x1,x2 is same ranking, x1 while be chose for x1 is before x2  
        out = Counter([i for i in max_freqs if i in ssvep_freqs]).most_common()
        if len(out)>1 and out[1][1]==out[0][1]:
            return False, out[0][0]
        else:
            return True, out[0][0]
    except:
        print(f'no ch found amplitude on certain freqs {ssvep_freqs}')
    return True, -1



def longestCommonSubsequence(text1, text2):
    m = len(text1)
    n = len(text2)
    if m * n == 0:
        return 0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max([dp[i - 1][j], dp[i][j - 1]])
    lcs = dp[m][n]

    array = np.array(dp)
    res = []
    posi = []
    posj = []
    l = lcs
    while l!=0:
        loc = np.argwhere(np.array(array)==l)
        for i,j in loc:
            if l-1 == array[i-1][j] and l-1 == array[i][j-1]:
                res.append(text1[i-1])
                l -= 1
                array = array[:i,:j]
                posi.append(i-1)
                posj.append(j-1)
                break

    if len(res) != dp[m][n]: raise
    return np.array(dp),dp[m][n],res[::-1],posi[::-1],posj[::-1]



import matplotlib.pyplot as plt
def AMPD(data,fp):
    """
    Parameters
    ----------
    data : 1-D numpy.ndarray 
    fp : 
    Returns
    ----------
    index list of the peaks
    """
    p_data = np.zeros_like(data, dtype=np.int32)
    count = data.shape[0]
    arr_rowsum = []
    for k in range(1, count // 2 + 1):
        row_sum = 0
        for i in range(k, count - k):
            if (fp and data[i] > data[i - k] and data[i] > data[i + k]) or\
                ( not fp and data[i] < data[i - k] and data[i] < data[i + k]) :
                row_sum += 1
        arr_rowsum.append(row_sum)
    max_index = np.argmax(arr_rowsum)
    max_window_length = max_index
    padding= max_window_length +1
    data_new = np.concatenate((np.full(padding, data[0]), data, np.full(padding, data[-1])))
    for k in range(1, max_window_length + 1):
        for i in range(padding, count + padding):
            if (fp and data_new[i] > data_new[i - k] and data_new[i] > data_new[i + k]) or\
                ( not fp and data_new[i] < data_new[i - k] and data_new[i] < data_new[i + k]) :
                p_data[i-padding] += 1
    x_peak = np.where(p_data == max_window_length)[0]
    return x_peak

from scipy.signal import find_peaks
from scipy.signal import find_peaks_cwt
def find_mypeaks(data, fp,figfile=None,ax=None,method='AMPD'):
    if method == 'AMPD':
        x_peak = AMPD(data,fp)
    elif method == 'scipy':
        if fp:
            x_peak, _ = find_peaks(data)
        else:
            x_peak, _ = find_peaks(-data)
    elif method == 'scipycwt':
        if fp:
            x_peak = find_peaks_cwt(data, widths=np.arange(1, 10))
        else:
            x_peak, _ = find_peaks(-data)
    else:
        raise

    if figfile or ax:
        if figfile:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        ax.plot(range(len(data)), data)
        ax.scatter(x_peak, data[x_peak], color="red")
        if figfile:
            plt.savefig(figfile,dpi=400)
            plt.close()

    return x_peak



import os
def append_content_onfile(file,content,method='a'):
    os.makedirs(os.path.dirname(os.path.abspath(file)), exist_ok=True)
    f = open(file,f"{method}+")
    f.write(content)
    f.close()



import numpy as np
def overview_of_data(data):
    data_np = np.array(data)
    data_shape = data_np.shape
    shape_dim = data_shape.index(max(data_shape))
    data_max = data_np.max(axis=shape_dim)
    data_min = data_np.min(axis=shape_dim)
    data_std = data_np.std(axis=shape_dim)
    return f'data of shape: {data_np.shape}, max: {data_max}, min: {data_min}, std: {data_std}'



