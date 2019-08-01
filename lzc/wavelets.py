import pywt
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from lzc.config import *


def read_data(raw, length=SPLIT_SIZE, max_len=MAX_LENGTH):
    raw_data = pd.read_csv(raw).iloc[:, 0].values
    raw_data = raw_data[:max_len]
    sure_value = math.floor(len(raw_data) / length) * length
    # print("sure of", sure_value, len(raw_data))
    # crop data
    raw_data = raw_data[:sure_value]
    # split data to length
    dds = np.array_split(raw_data, (len(raw_data) / length))
    return dds, raw_data


def plot(y,title =""):
    plt.title(title)
    x = np.linspace(0, len(y) - 1, len(y))
    plt.plot(x, y)
    plt.show()


def get_transformed(data, func):
    retCA = []
    retCD = []
    for i in data:
        # print(len(i), "Fuck!")
        cA = np.pad(cA, (0, len(i) - len(cA)), mode='constant')
        cD = np.pad(cD, (0, len(i) - len(cD)), mode='constant')
        retCA = retCA + cA.tolist()
        retCD = retCD + cD.tolist()

    return retCA, retCD


def plot_each(data, func):
    (cA, cD) = pywt.dwt(data[0], func)
    plot(cA,'cA of DWTUnit('+func+")")
    plot(cD,'cD of DWTUnit('+func+")")


def to_wavs(fname, max_len=MAX_LENGTH, attr='csv'):
    datas, rd = read_data(fname + "." + attr, max_len=max_len)
    df = pd.DataFrame()
    df["basic"] = rd
    for i in WAVELETS:
        print(i)
        ca, cd = get_transformed(datas, i)
        df[i + "_cA"] = ca
        df[i + "_cD"] = cd
    df.to_csv(fname + "_dwt300.csv", float_format='%.3f')

def show_wav(fname, max_len = MAX_LENGTH, attr='csv'):
    datas, rd = read_data(fname + "." + attr, max_len=max_len)
    plot(datas[0],'input')
    for i in WAVELETS:
        plot_each(datas,i)

if __name__ == '__main__':
    # to_wavs("olddata/m0", max_len=OLD_DATA_LEN, attr='txt')
    # to_wavs("olddata/m1", max_len=OLD_DATA_LEN, attr='txt')
    # to_wavs("olddata/m2", max_len=OLD_DATA_LEN, attr='txt')
    # to_wavs('0m')
    # to_wavs('1m')
    # to_wavs('2m')
    # print(len(pywt.wavelist(kind='discrete')))
    # for i in pywt.wavelist(kind='discrete'):
    #     print(i)
    show_wav('1m')
