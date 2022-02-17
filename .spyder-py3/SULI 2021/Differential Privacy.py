import pandas as pd
import matplotlib.pylab as plt
import matplotlib.pyplot as plot
import matplotlib.patches as patches
import numpy as np



count = 0
list_num = []
row_count = 0
highest = 0
lowest = 10,000
cols_to_use = ['Household 1']
df = pd.read_excel(r"C:\Users\marti\Documents\Residential-Profiles.xlsx", sheet_name = 'Residential-Profiles.csv', index_col = False, usecols = cols_to_use, nrows = 74) 

num_rows = len(df['Household 1'])

clean_signal = df
mu, sigma = 0, 100
noise = np.random.normal(mu, sigma, [num_rows, len(cols_to_use)])
signal = clean_signal + noise

difference = signal - clean_signal


        
for row in df['Household 1']:
       list_num.append(count * 10 / 60)
       count+=1
           





#for column in df.columns:
 #   for row in df['Household 1']:
  #      if df[column][row_count] == signal[column][row_count]:
   #         list_num.append(count)
    #        count += 1
            
   # row_count += 1
   

plot.plot(list_num, clean_signal)
plot.plot(list_num, signal)
plot.xlabel("Time (hr)")
plot.ylabel("kW/hr")
plot.legend(['Original Data', 'Noisy Data'])

