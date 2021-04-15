import numpy as np
import matplotlib.pyplot as plt
from data_by_region import data_by_region
from temperature import temperature, month

PLOT = True
SAVEFIG = True
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
    'AGE 0-4': data_by_region(REGION, 'AGE 0-4', RANGE),
    'AGE 5-24': data_by_region(REGION, 'AGE 5-24', RANGE),
    'AGE 25-49': data_by_region(REGION, 'AGE 25-49', RANGE),
    'AGE 50-64': data_by_region(REGION, 'AGE 50-64', RANGE),
    'AGE 65': data_by_region(REGION, 'AGE 65', RANGE),
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

regions = {
    'New England': data_by_region('New England', 'TOTAL PATIENTS', RANGE),
    'Mid-Atlantic': data_by_region('Mid-Atlantic', 'TOTAL PATIENTS', RANGE),
    'East North Central': data_by_region('East North Central', 'TOTAL PATIENTS',
                                         RANGE),
    'West North Central': data_by_region('West North Central', 'TOTAL PATIENTS',
                                         RANGE),
    'South Atlantic': data_by_region('South Atlantic', 'TOTAL PATIENTS', RANGE),
    'East South Central': data_by_region('East South Central', 'TOTAL PATIENTS',
                                         RANGE),
    'West South Central': data_by_region('West South Central', 'TOTAL PATIENTS',
                                         RANGE),
    'Mountain': data_by_region('Mountain', 'TOTAL PATIENTS', RANGE),
    'Pacific': data_by_region('Pacific', 'TOTAL PATIENTS', RANGE),
}

region_heatmap = np.zeros((len(regions), len(regions)))
for group1 in regions.keys():
    for group2 in regions.keys():
        cross_correlation = np.correlate(
            regions[group1] / np.linalg.norm(regions[group1]),
            regions[group2] / np.linalg.norm(regions[group2]),
            'full')
        region_heatmap[
            np.where(np.array(list(regions.keys())) == group1)[0][0],
            np.where(np.array(list(regions.keys())) == group2)[0][0]
        ] = max(cross_correlation)

if PLOT:
    plt.figure()
    plt.plot(month, temperature)
    plt.xlabel('# of months after January 1998')
    plt.ylabel('$^{\circ}F$')
    # plt.title(
    #     f"Average monthly temperatures in the {REGION} region from January 1998 to December 2014",
    #     fontsize=8)
    plt.title(
        f"Average monthly temperatures from January 1998 to December 2014")
    if SAVEFIG:
        plt.savefig('8')

    plt.figure()
    plt.plot(month, temperature_change * 100, label='Temperature')
    plt.plot(month, patient_change * 100, label='Total number of patients')
    plt.legend()
    plt.xlabel('# of months after January 1998')
    plt.ylabel('% change from previous month')
    plt.title("Percent change in temperature and total number of patients")
    if SAVEFIG:
        plt.savefig('9')

    plt.figure()
    plt.xcorr(patient_change, -temperature_change, normed=False, maxlags=None)
    plt.xlabel('Lag [month]')
    plt.ylabel('Correlation')
    plt.title(
        "Cross-correlation between patient number changes and negative temperature changes",
        fontsize=10)
    if SAVEFIG:
        plt.savefig('10')

    plt.figure()
    plt.xcorr(np.sign(patient_change), np.sign(-temperature_change),
              normed=False, maxlags=None)
    plt.xlabel('Lag [month]')
    plt.ylabel('Correlation')
    plt.title(
        "Bit-converted cross-correlation between patient number changes and negative temperature changes",
        fontsize=8)
    if SAVEFIG:
        plt.savefig('11')

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
            text = ax.text(j, i, round(corr_heatmap[i, j], 3),
                           ha="center", va="center", color="r")
    ax.set_title("Age group correlation map")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if SAVEFIG:
        plt.savefig('12')

    fig, ax = plt.subplots()
    im = ax.imshow(region_heatmap, cmap='gray')
    ax.set_xticks(np.arange(len(regions)))
    ax.set_yticks(np.arange(len(regions)))
    ax.set_xticklabels(list(regions.keys()))
    ax.set_yticklabels(list(regions.keys()))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(len(regions.keys())):
        for j in range(len(regions.keys())):
            text = ax.text(j, i, round(region_heatmap[i, j], 2),
                           ha="center", va="center", color="r")
    ax.set_title("Region correlation map")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if SAVEFIG:
        plt.savefig('13')

    plt.show()
