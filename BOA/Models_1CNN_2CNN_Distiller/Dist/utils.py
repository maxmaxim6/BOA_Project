import time
#from matplotlib.pyplot import axis
import pandas as pd
from sklearn.model_selection import KFold,train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score,recall_score,precision_score
import numpy as np
import ast
import tkinter as tk

def current_time():
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    return current_time

# shuffle the model and split k fold
def load_data(file_path,preprocessing,label_columns,remove_columns=None,k=5,shuffle=True,random_state=42):
    label_columns=['label']
    df = pd.read_csv(file_path,index_col=None,header=0)
    df = preprocessing(df)
    if shuffle:
        df = df.sample(frac=1, random_state=random_state)


    fold = KFold(k,shuffle=shuffle,random_state=random_state)
    for train_index, test_index in fold.split(df):
        yield df.iloc[train_index].drop(label_columns,axis=1),df.iloc[train_index][label_columns],df.iloc[test_index].drop(label_columns,axis=1),df.iloc[test_index][label_columns]

# prepare the data, clear data.
def distiller_preprocessing(df):
    df['udps.n_bytes'] = df['udps.n_bytes'].apply(ast.literal_eval)
    df['udps.n_bytes'] = df['udps.n_bytes'].apply(np.asarray,dtype='float16')
    df['udps.protocol_header_fields'] = df['udps.protocol_header_fields'].apply(ast.literal_eval)
    df['udps.stnn_image'] = df['udps.stnn_image'].apply(ast.literal_eval)
    df['udps.stnn_image'] = df['udps.stnn_image'].apply(np.asarray,dtype='float32')
    enc = OneHotEncoder(handle_unknown='ignore')
    df['label'] = list(enc.fit_transform(df['label'].values.reshape(-1,1)).toarray())
    print(enc.categories_)
    return df

    
