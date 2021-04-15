import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLOT = True
FILE = 'influenza.csv'  # https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/29023
REGION = 'Pacific'
RANGE = [14, -3]  # from 1998-01 to 2014-12
FIT_ORDER = 10

data = pd.read_csv(FILE, skiprows=1)
data = data[data['REGION'] == REGION][RANGE[0]:RANGE[1]]
# print(data['REGION'].unique())

total_patients = data['TOTAL PATIENTS'].to_numpy()
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


trend = fit(week, total_patients, FIT_ORDER)

ft_freq = np.fft.fftshift(np.fft.fftfreq(len(week), 7 / 365))
ft_raw = np.fft.fftshift(np.fft.fft(total_patients))
ft_cleaned = np.fft.fftshift(np.fft.fft(total_patients - trend))

boxcar_filter = np.ones(len(ft_freq))
hann_filter = np.zeros(len(ft_freq))
for i in range(len(ft_freq)):
    if abs(ft_freq[i]) > 1:
        boxcar_filter[i] = 0
    hann_filter[i] = np.sin(np.pi * i / (len(ft_freq) - 1)) ** 2

ft_filtered = ft_cleaned * boxcar_filter

if PLOT and __name__ == '__main__':
    plt.figure()
    plt.plot(week, total_patients)
    plt.xlabel('# of weeks after January 1998')
    plt.ylabel('# of total patients')
    plt.title(
        "Total number of patients from Influenza-like-illnesses in the United States from January 1998 to December 2014",
        fontsize=8)

    plt.figure()
    plt.plot(week, trend)
    plt.xlabel('# of weeks after January 1998')
    plt.ylabel('# of total patients')
    plt.title(f"Polynomial fit of total patient numbers to the {FIT_ORDER}th order")

    plt.figure()
    plt.plot(week, total_patients - trend)

    plt.figure()
    plt.stem(ft_freq, np.square(np.abs(ft_cleaned)), use_line_collection=True)

    plt.figure()
    plt.stem(ft_freq, np.square(np.abs(ft_filtered)), use_line_collection=True)

    plt.figure()
    plt.plot(week, np.fft.ifft(np.fft.ifftshift(ft_filtered)))

    plt.figure()
    plt.plot(week, total_patients)
    plt.plot(week, np.fft.ifft(np.fft.ifftshift(ft_filtered)) + trend)

    plt.show()
