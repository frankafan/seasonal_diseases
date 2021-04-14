import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE = 'west_climate_region_temp.csv'
RANGE = [12, -13]  # from 1998-01 to 2014-12
data = pd.read_csv(FILE, skiprows=4)[RANGE[0]:RANGE[1]]

temperature = data['Value'].to_numpy()
month = np.arange(len(temperature))

plt.figure()
plt.plot(month, temperature)
plt.show()
