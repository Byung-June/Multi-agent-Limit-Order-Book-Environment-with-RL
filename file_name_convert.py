import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


current_dir = os.getcwd()
current_dir += '/log'
learning_curve_list = glob.glob(current_dir + '/apex*.csv')

for file_name in learning_curve_list:
    if len(file_name.split('__')) == 2:
        try:
            x, y = file_name.split('__')
            after = x.split('train')[0] + x[-3:] + '_train_' + y
            os.rename(file_name, after)
            print(file_name, after)
        except FileExistsError:
            print('Delete!', file_name)
            os.remove(file_name)
print('DONE!')
