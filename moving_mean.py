import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 50
df = pd.read_csv('signal.txt', sep="\n", header=None)

new_signal = df.rolling(N).mean()
y1 = list(df.values.flatten())
y2 = list(new_signal.values.flatten())

plt.plot(y1, color="b", alpha=0.4)
plt.plot(y2, color="b")
plt.show()
