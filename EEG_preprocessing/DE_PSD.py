import os
import numpy as np
import math
import scipy.io as sio
from scipy.fftpack import fft, ifft


def DE_PSD(data, fre, time_window, which="both"):
    """Compute Differential Entropy (DE) and/or Power Spectral Density (PSD).

    Parameters
    ----------
    data : np.ndarray
        Array of shape ``(n_channels, n_samples)`` containing the EEG segment.
    fre : int
        Sampling frequency of ``data``.
    time_window : float
        Window length in seconds used for the STFT.
    which : {"both", "de", "psd"}
        Selects which features to compute.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        Depending on ``which`` either DE, PSD or both.
    """
    #initialize the parameters
    # STFTN=stft_para['stftn']
    # fStart=stft_para['fStart']
    # fEnd=stft_para['fEnd']
    # fs=stft_para['fs']
    # window=stft_para['window']
    
    STFTN = 200
    fStart = [1, 4, 8, 14, 31]
    fEnd = [4, 8, 14, 31, 99] # bands : delta, theta, alpha, beta, gamma
    window = time_window
    fs = fre

    WindowPoints=fs*window

    fStartNum=np.zeros([len(fStart)],dtype=int)
    fEndNum=np.zeros([len(fEnd)],dtype=int)
    for i in range(0,len(fStart)):
        fStartNum[i]=int(fStart[i]/fs*STFTN)
        fEndNum[i]=int(fEnd[i]/fs*STFTN)

    #print(fStartNum[0],fEndNum[0])
    n=data.shape[0]
    m=data.shape[1]

    #print(m,n,l)
    if which in ("both", "psd"):
        psd = np.zeros((n, len(fStart)), dtype=float)
    else:
        psd = None

    if which in ("both", "de"):
        de = np.zeros((n, len(fStart)), dtype=float)
    else:
        de = None
    #Hanning window
    Hlength=int(window*fs) ##added int()
    #Hwindow=hanning(Hlength)
    Hwindow= np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (Hlength+1)) for n in range(1,Hlength+1)])

    WindowPoints = fs * window
    dataNow = data[0:n]
    for j in range(n):
        temp = dataNow[j]
        Hdata = temp * Hwindow
        FFTdata = fft(Hdata, STFTN)
        magFFTdata = abs(FFTdata[0 : int(STFTN / 2)])
        for p in range(len(fStart)):
            E = 0
            for p0 in range(fStartNum[p] - 1, fEndNum[p]):
                E += magFFTdata[p0] * magFFTdata[p0]
            E = E / (fEndNum[p] - fStartNum[p] + 1)
            if psd is not None:
                psd[j][p] = E
            if de is not None:
                de[j][p] = math.log(100 * E, 2)
    
    if which == "de":
        return de
    if which == "psd":
        return psd
    return de, psd

