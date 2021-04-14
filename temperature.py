import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE = 'west_climate_region_temp.csv'
RANGE = [8, -12]
data = pd.read_csv(FILE, skiprows=4)[RANGE[0]:RANGE[1]]

print(data)
