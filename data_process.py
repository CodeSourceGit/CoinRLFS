import numpy as np
import pandas as pd


def ReadData(name):
    if name == 'arrhythmia':
        dataset = pd.read_csv('dataset/arrhythmia.csv')
        return dataset  
    elif name == 'spam':
        dataset = pd.read_csv('dataset/spam.csv')
        return dataset
    return None


