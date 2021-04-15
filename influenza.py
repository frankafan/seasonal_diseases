import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLOT = True
SAVEFIG = True
FILE = 'influenza.csv'  # https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/29023
REGION = 'Pacific'
RANGE = [14, -3]  # from 1998-01 to 2014-12
FIT_ORDER = 10

data = pd.read_csv(FILE, skiprows=1)
# print(data['REGION'].unique())
data = data[data['REGION'] == REGION][RANGE[0]:RANGE[1]]

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
gaussian_filter = np.zeros(len(ft_freq))
for i in range(len(ft_freq)):
    if abs(ft_freq[i]) > 1:
        boxcar_filter[i] = 0
    gaussian_filter[i] = np.exp(- np.pi ** 2 * ft_freq[i] ** 2 / 15)

ft_boxcar_filtered = ft_cleaned * boxcar_filter
ft_gaussian_filtered = ft_cleaned * gaussian_filter

if PLOT and __name__ == '__main__':
    plt.figure()
    plt.plot(week, total_patients)
    plt.xlabel('# of weeks after January 1998')
    plt.ylabel('# of total patients')
    # plt.title(
    #     f"Total number of patients from Influenza-like-illnesses in the United States {REGION} region from January 1998 to December 2014",
    #     fontsize=7)
    plt.title(
        f"Total number of patients from Influenza-like-illnesses in the United States from January 1998 to December 2014",
        fontsize=8)
    if SAVEFIG:
        plt.savefig('1')

    plt.figure()
    plt.plot(week, total_patients, label='Original data')
    plt.plot(week, trend, label='Fit curve')
    plt.legend()
    plt.xlabel('# of weeks after January 1998')
    plt.ylabel('# of total patients')
    plt.title(
        f"Polynomial fit of total patient numbers to the {FIT_ORDER}th order")
    if SAVEFIG:
        plt.savefig('2')

    plt.figure()
    plt.plot(week, total_patients - trend)
    plt.xlabel('# of weeks after January 1998')
    plt.ylabel('# of patients deviating from fit curve')
    plt.title("De-trended time series")
    if SAVEFIG:
        plt.savefig('3')

    plt.figure()
    plt.plot(ft_freq, np.square(np.abs(ft_cleaned)))
    plt.xlabel('$\omega [year^{-1}]$')
    plt.ylabel('Intensity')
    plt.title("Power spectrum of de-trended time series")
    if SAVEFIG:
        plt.savefig('4')

    plt.figure()
    plt.stem(ft_freq, np.square(np.abs(ft_cleaned)), use_line_collection=True)
    plt.xlim([0, 5])
    plt.xlabel('$\omega [year^{-1}]$')
    plt.ylabel('Intensity')
    plt.title("Power spectrum of de-trended time series")
    if SAVEFIG:
        plt.savefig('5')

    plt.figure()
    plt.stem(ft_freq, np.square(np.abs(ft_boxcar_filtered)), use_line_collection=True)
    plt.xlim([0, 5])
    plt.xlabel('$\omega [year^{-1}]$')
    plt.ylabel('Intensity')
    plt.title("Filtered power spectrum of de-trended time series")
    if SAVEFIG:
        plt.savefig('6')

    plt.figure()
    plt.plot(week, total_patients, label='Original data')
    plt.plot(week, np.fft.ifft(np.fft.ifftshift(ft_boxcar_filtered)) + trend,
             label='Filtered data')
    plt.legend()
    plt.xlabel('# of weeks after January 1998')
    plt.ylabel('# of total patients')
    plt.title("Filtered time series")
    if SAVEFIG:
        plt.savefig('7')

    # plt.figure()
    # plt.plot(week, total_patients, label='Original data')
    # plt.plot(week, np.fft.ifft(np.fft.ifftshift(ft_boxcar_filtered)) + trend, label='Boxcar-filtered data')
    # plt.plot(week, np.fft.ifft(np.fft.ifftshift(ft_gaussian_filtered)) + trend, label='Gaussian-filtered data')
    # plt.legend()
    # plt.xlabel('# of weeks after January 1998')
    # plt.ylabel('# of total patients')
    # plt.title("Filtered time series")

    plt.show()
