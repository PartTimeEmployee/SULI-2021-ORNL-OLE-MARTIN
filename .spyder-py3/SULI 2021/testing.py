import pandas as pd
import matplotlib.pylab as pl
import matplotlib.patches as patches
import numpy as np



count = 0

row_count = 0

cols_to_use = ['Household 1', 'Household 2', 'Household 3', 'Household 4', 'Household 5', 'Household 6', 'Household 7', 'Household 8', 'Household 9', 'Household 10', 'Household 11', 'Household 12', 'Household 13', 'Household 14', 'Household 15']
df = pd.read_excel(r"C:\Users\marti\Documents\Residential-Profiles.xlsx", sheet_name = 'Residential-Profiles.csv', index_col = False, usecols = cols_to_use) 

num_rows = len(df['Household 1'])

clean_signal = df
mu, sigma = 0, 100
noise = np.random.normal(mu, sigma, [num_rows, len(cols_to_use)])
signal = clean_signal + noise

for column in df.columns:
    for row in df['Household 1']:
        if df[column][row_count] == signal[column][row_count]:
            count += 1
    row_count += 1





