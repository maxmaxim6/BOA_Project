import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import time


###################
# Model utilities #
###################

def stack(layers):
    '''
    Using the Functional-API of Tensorflow to build a sequential
    network (stacked layers) from list of layers.
    '''
    layer_stack = None
    for layer in layers:
        if layer_stack is None:
            layer_stack = layer
        else:
            layer_stack = layer(layer_stack)
    return layer_stack


###############################
# Report/Experiment utilities #
###############################

class KFoldClassificationReport():
    def __init__(self) -> None:
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1s = []
        self.cf_matrices = []
        self.cm_matrices = []
        
    def add(self, acc, prec, recall, f1, cf_matrix):
        self.accuracies.append(acc)
        self.precisions.append(prec)
        self.recalls.append(recall)
        self.f1s.append(f1)
        self.cf_matrices.append(cf_matrix)
        
    def print(self):
        print(self.report_text())

    def export_to_file(self, filepath):
        f = open(filepath, "w")
        f.write(self.report_text())
        f.close()

    def report_text(self):
        return '\n'.join([
            '------------ Classifiation Report ------------',
            ' '.join(['Accuracy: ', self.string_of_mean_std(np.mean(self.accuracies), np.std(self.accuracies))]),
            ' '.join(['Precision:', self.string_of_mean_std(np.mean(self.precisions), np.std(self.precisions))]),
            ' '.join(['Recall:   ', self.string_of_mean_std(np.mean(self.recalls), np.std(self.recalls))]),
            ' '.join(['F1:       ', self.string_of_mean_std(np.mean(self.f1s), np.std(self.f1s))]),
            '---- Confusion Matrix ----',
            self.string_normalized_cf_matrix(sum(self.cf_matrices, np.zeros(self.cf_matrices[0].shape))/len(self.cf_matrices))
        ])

    def string_of_mean_std(self, mean, std):
        return "{:.4f}".format(mean) + ' (+- '+"{:.4f}".format(std)+')'

    def string_normalized_cf_matrix(self, cf_matrix):
        normalized_cf_matrix = np.apply_along_axis(lambda r: r/np.sum(r), axis=1, arr=cf_matrix)
        return str(normalized_cf_matrix)

def current_time():
    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y-%H-%M-%S", t)
    return current_time

#####################################
# Data loading/preprocess utilities #
#####################################

def load_data(file_path, preprocessing, label_column,columns, k=5, shuffle=True, random_state=42):
    df = pd.read_csv(file_path,index_col=None,header=0,usecols=columns)
    df = preprocessing(df)
    if shuffle:
        df = df.sample(frac=1, random_state=random_state)
    # print(df.head())
    # print(df.memory_usage(index=True, deep=True))
    fold = KFold(k, shuffle=shuffle, random_state=random_state)
    for train_index, test_index in fold.split(df):
        yield df.iloc[train_index].drop(label_column,axis=1),df.iloc[train_index][label_column],df.iloc[test_index].drop(label_column,axis=1),df.iloc[test_index][label_column]


def preprocessing_m1cnn_classification(df):
    df['udps.n_bytes'] = df['udps.n_bytes'].apply(ast.literal_eval)
    df['udps.n_bytes'] = df['udps.n_bytes'].apply(np.array, dtype='float32')
    enc = OneHotEncoder(handle_unknown='ignore')
    df['label'] = list(enc.fit_transform(df['label'].values.reshape(-1,1)).toarray())
    return df



