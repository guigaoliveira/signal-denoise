import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pywt
import seaborn
from statsmodels.robust import mad

df = pd.read_csv('signal.txt', sep="\n", header=None)

signal = np.array(df.values.flatten())
print(signal)


def waveletSmooth(x, wavelet="db4", level=4, title=None):
    # calculate the wavelet coefficients
    coeff = pywt.wavedec(x, wavelet, mode="per")
    # calculate a threshold
    sigma = mad(coeff[-level])
    # changing this threshold also changes the behavior,
    # but I have not played with this very much
    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft")
                 for i in coeff[1:])
    # reconstruct the signal using the thresholded coefficients
    y = pywt.waverec(coeff, wavelet, mode="per")
    f, ax = plt.subplots()
    plt.plot(x, color="b", alpha=0.5)
    plt.plot(y, color="b")
    if title:
        ax.set_title(title)
    ax.set_xlim((0, len(y)))
    plt.show()


waveletSmooth(signal)
