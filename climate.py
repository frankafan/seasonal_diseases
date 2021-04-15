import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TEMPERATURE_FILE = 'national_average_temperature_1997-2016.csv'  # https://www.ncdc.noaa.gov/cag/national/time-series
PRECIPITATION_FILE = 'national_precipitation_1997-2016.csv'
RANGE = [12, -13]  # from 1998-01 to 2014-12
temp_data = pd.read_csv(TEMPERATURE_FILE, skiprows=4)[RANGE[0]:RANGE[1]]
prec_data = pd.read_csv(PRECIPITATION_FILE, skiprows=4)[RANGE[0]:RANGE[1]]

temperature = temp_data['Value'].to_numpy()
precipitation = prec_data['Value'].to_numpy()
month = np.arange(len(temperature))

if __name__ == '__main__':
    plt.figure()
    plt.plot(month, temperature)
    plt.figure()
    plt.plot(month, precipitation)
    plt.show()
