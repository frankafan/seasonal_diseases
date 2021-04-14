import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLOT = True
FILE = 'influenza.csv'
REGION = 'Pacific'
RANGE = [0, -1]

data = pd.read_csv(FILE, skiprows=1)
data = data[data['REGION'] == REGION]

total_patients = data['TOTAL PATIENTS'].to_numpy()[RANGE[0]:RANGE[1]]
week = np.arange(len(total_patients))
for i in range(len(total_patients)):
    if np.isnan(total_patients[i]) and (i > 0):
        total_patients[i] = total_patients[i - 1]


def fit(time, data, degree):
    """Return the y values of the fit curve"""
    y = []
    fit_coefficients = np.polyfit(time.astype(float), data, degree)
    for t in time.astype(float):
        y_value = 0
        for d in range(degree + 1):
            y_value += fit_coefficients[d] * t ** (degree - d)
        y.append(y_value)
    return np.array(y)


trend = fit(week, total_patients, 20)

ft_freq = np.fft.fftshift(np.fft.fftfreq(len(week), 1))
ft_raw = np.fft.fftshift(np.fft.fft(total_patients))
ft_cleaned = np.fft.fftshift(np.fft.fft(total_patients - trend))

ft_filtered = ft_cleaned.copy()
for i in range(len(ft_freq)):
    if abs(ft_freq[i]) > 0.007:
        ft_filtered[i] = 0
    if abs(ft_cleaned[i]) ** 2 < 0.1e10:
        ft_filtered[i] = 0

if PLOT:
    plt.figure()
    plt.plot(week, total_patients)

    plt.figure()
    plt.plot(week, trend)

    plt.figure()
    plt.plot(week, total_patients - trend)

    plt.figure()
    plt.stem(ft_freq, np.square(np.abs(ft_raw)), use_line_collection=True)

    plt.show()
