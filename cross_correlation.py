import numpy as np
import matplotlib.pyplot as plt
from influenza import total_patients
from temperature import temperature, month

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
plt.xcorr(patient_change, -temperature_change)
plt.show()
