import numpy as np
import matplotlib.pyplot as plt
from data_by_region import data_by_region
from temperature import temperature, month

REGION = 'Pacific'
PROPERTY = 'TOTAL PATIENTS'
RANGE = [14, -3]  # from 1998-01 to 2014-12

total_patients = data_by_region(REGION, PROPERTY, RANGE)
patients_monthly = total_patients[::4][:-18]
temperature_change = np.zeros(len(temperature))
patient_change = np.zeros(len(temperature))
for i in range(1, len(temperature)):
    temperature_change[i] = (temperature[i] - temperature[
        i - 1]) / temperature[i - 1]
    patient_change[i] = (patients_monthly[i] - patients_monthly[
        i - 1]) / patients_monthly[i - 1]

plt.figure()
plt.plot(month, temperature_change)
plt.plot(month, patient_change)
plt.figure()
plt.xcorr(patient_change, temperature_change, maxlags=None)
plt.figure()
plt.xcorr(np.sign(patient_change), np.sign(temperature_change), maxlags=None)
plt.show()
