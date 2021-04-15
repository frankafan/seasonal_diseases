import numpy as np
import matplotlib.pyplot as plt
from data_by_region import data_by_region
from temperature import temperature, month

PLOT = True
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

age_groups = {
    'age0-4': data_by_region(REGION, 'AGE 0-4', RANGE),
    'age5-24': data_by_region(REGION, 'AGE 5-24', RANGE),
    'age25-40': data_by_region(REGION, 'AGE 25-49', RANGE),
    'age50-64': data_by_region(REGION, 'AGE 50-64', RANGE),
    'age65-over': data_by_region(REGION, 'AGE 65', RANGE),
}

corr_heatmap = np.zeros((len(age_groups), len(age_groups)))
for group1 in age_groups.keys():
    for group2 in age_groups.keys():
        cross_correlation = np.correlate(
            age_groups[group1] / np.linalg.norm(age_groups[group1]),
            age_groups[group2] / np.linalg.norm(age_groups[group2]),
            'full')
        corr_heatmap[
            np.where(np.array(list(age_groups.keys())) == group1)[0][0],
            np.where(np.array(list(age_groups.keys())) == group2)[0][0]
        ] = max(cross_correlation)

if PLOT:
    plt.figure()
    plt.plot(month, temperature_change * 100, label='Temperature')
    plt.plot(month, patient_change * 100, label='Total number of patients')
    plt.legend()
    plt.xlabel('# of months after January 1998')
    plt.ylabel('% change from previous month')
    plt.title("Percent change in temperature and total number of patient")

    plt.figure()
    plt.xcorr(patient_change, -temperature_change, maxlags=None)

    plt.figure()
    plt.xcorr(np.sign(patient_change), np.sign(-temperature_change),
              maxlags=None)

    fig, ax = plt.subplots()
    im = ax.imshow(corr_heatmap, cmap='gray')

    ax.set_xticks(np.arange(len(age_groups)))
    ax.set_yticks(np.arange(len(age_groups)))
    ax.set_xticklabels(list(age_groups.keys()))
    ax.set_yticklabels(list(age_groups.keys()))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(len(age_groups.keys())):
        for j in range(len(age_groups.keys())):
            text = ax.text(j, i, round(corr_heatmap[i, j], 2),
                           ha="center", va="center", color="r")
    ax.set_title("Age group correlation map")
    fig.tight_layout()
    plt.show()
