import numpy as np
import pandas as pd

FILE = 'influenza.csv'  # https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/29023


def data_by_region(region, data_property, data_range):
    data = pd.read_csv(FILE, skiprows=1)
    data = data[data['REGION'] == region][data_range[0]:data_range[1]]
    total_patients = data[data_property].to_numpy()
    for i in range(len(total_patients)):
        if np.isnan(total_patients[i]) and (i > 0):
            total_patients[i] = total_patients[i - 1]
    return total_patients


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
