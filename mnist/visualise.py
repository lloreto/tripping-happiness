#Visualise mnist data
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

train_data = pd.read_csv(r'data/train.csv')
averaged = train_data.groupby('label').mean()

#convert to images
f, axes = plt.subplots(2,5,sharey='all',sharex='all')
for idx,img in enumerate(averaged.as_matrix()):
    ax = axes[np.unravel_index(idx,(2,5))] 
    ax.imshow(img.reshape(28,28),cmap=plt.get_cmap('gray_r'))

