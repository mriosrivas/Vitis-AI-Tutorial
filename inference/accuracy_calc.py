#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



def fix2float(data, bit_width = 8, fix_point = 7):
    scale = 2**(-fix_point)
    return (np.maximum(-np.power(2, bit_width - 1), 
                       np.minimum(data, np.power(2, bit_width - 1) - 1)))*scale



df = pd.read_csv('result.csv')


unique = list(np.sort(df['Label'].unique()))
min_val = df['Label'].min()
max_val = df['Label'].max()


df['Label'] = df['Label'].apply(fix2float)


unique = list(np.sort(df['Label'].unique()))


df['prediction'] = (df['Label']>0.5).astype('int')
df['ground_truth'] = df['ID'].str.split('.', expand=True)[0]
df['int_ground_truth'] = (df['ground_truth'] == 'dog').astype('int')

cm = confusion_matrix(df['int_ground_truth'], df['prediction'], normalize='true')
tn, fp, fn, tp = cm.ravel()
accuracy = np.round(accuracy_score(df['int_ground_truth'], df['prediction']), 1)


print(f'accuracy = {accuracy}')

