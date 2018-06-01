import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.robust import mad
import pywt

df = pd.read_csv('signal.txt', sep="\n", header=None)
signal = np.array(df.values.flatten())

def waveletDenoising(x, wavelet="db4", level=3):
    # calculate the wavelet coefficients
    coeff = pywt.wavedec(x, wavelet, mode="per")
    # calculate a threshold
    sigma = mad(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft")
                 for i in coeff[1:])
    # reconstruct the signal using the thresholded coefficients
    y = pywt.waverec(coeff, wavelet, mode="per")
    fig, ax = plt.subplots()
    plt.plot(x, color="b", alpha=0.4)
    plt.plot(y, color="b")
    ax.set_title('Signal Denoising')
    ax.set_xlim((0, len(y)))
    plt.show()


waveletDenoising(signal)
