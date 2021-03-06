import numpy as np
import pandas as pd

FILE = 'influenza.csv'  # https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/29023


def data_by_region(region, data_property, data_range):
    data = pd.read_csv(FILE, skiprows=1)
    data = data[data['REGION'] == region][data_range[0]:data_range[1]]
    cases = data[data_property].to_numpy()
    for i in range(len(cases)):
        if cases[i] == 'X' or np.isnan(np.float(cases[i])):
            if i > 0:
                cases[i] = cases[i - 1]
            else:
                cases[i] = 0
        cases[i] = np.float(cases[i])
    return cases


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


def filter_time_series(time_series, dt, w):
    ft_freq = np.fft.fftshift(np.fft.fftfreq(len(time_series), dt))
    ft = np.fft.fftshift(np.fft.fft(time_series))

    ft_filtered = ft.copy()
    for i in range(len(ft_freq)):
        if abs(ft_freq[i]) > w:
            ft_filtered[i] = 0
    series_filtered = np.fft.ifft(np.fft.ifftshift(ft_filtered))
    return series_filtered
