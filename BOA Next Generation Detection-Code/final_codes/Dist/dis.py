from model_stnn_dis import Distiller
import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from sklearn.model_selection import KFold,train_test_split
import time
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score,recall_score,precision_score,classification_report
import seaborn as sns
import os
from utils import current_time,load_data,distiller_preprocessing
from tensorflow.keras.utils import to_categorical

# report 
class KFoldClassificationReport():
    def __init__(self) -> None:
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1s = []
        self.cf_matrices = []
        
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



#  define labels and features 
col_le=['label']
col_fe=['udps.n_bytes','udps.protocol_header_fields','udps.stnn_image','label']
report = KFoldClassificationReport()
input_filenames= r'C:\Users\maxma\Desktop\PROJECT\Codes\fs_data_filter.csv'

#  load data and run the algorithm 
for x_train,y_train,x_test,y_test in load_data(input_filenames,distiller_preprocessing,'label',col_fe):


# define number of labels and initialize the model 
    d = Distiller(25)



    print('#####################STNN-PreTraining##########################')
    pretraining_model = d.get_model_for_pretraining(d.stnn_model)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    pretraining_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss=loss_fn,
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5,min_delta=0.01)
    pretraining_model.fit(np.stack(x_train['udps.stnn_image']),
                        [np.stack(y_train)],
                        epochs=30,      
                        verbose=2,batch_size=50,callbacks=[callback],workers=4,use_multiprocessing=False)



    print('#####################n-bytes-PreTraining##########################')
    pretraining_model = d.get_model_for_pretraining(d.payload_model)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    pretraining_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss=loss_fn,
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=0.01)
    pretraining_model.fit(np.stack(x_train['udps.n_bytes']),
                            [np.stack(y_train)],
                            epochs=30,
                            verbose=2,batch_size=50,callbacks=[callback],workers=4,use_multiprocessing=False)



    print('#####################HEADER-Fields-PreTraining##########################')
    pretraining_model = d.get_model_for_pretraining(d.proto_model)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    pretraining_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
    loss=loss_fn,
    metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5,min_delta=0.01)
    pretraining_model.fit(np.stack(x_train['udps.protocol_header_fields']),
                        [np.stack(y_train)],
                        epochs=30,
                        verbose=2,batch_size=50,callbacks=[callback],workers=4,use_multiprocessing=False)

    print('#####################FINE-TUNING##########################')
    d.freeze_for_finetuning()

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    d.model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=loss_fn,
    metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10,min_delta=0.01)
    d.model.fit([np.stack(x_train['udps.n_bytes']), 
                np.stack(x_train['udps.protocol_header_fields']),
                np.stack(x_train['udps.stnn_image'])],
                [np.stack(y_train)],
        epochs=40, 
        verbose=2,
        batch_size=50,
        callbacks=[callback],
        workers=4,
        use_multiprocessing=False)
    end = time.time()



#  predictions 
    stack_x_test = [np.stack(x_test['udps.n_bytes']),
                    np.stack(x_test['udps.protocol_header_fields']),
                    np.stack(x_test['udps.stnn_image'])]


    print('[Status] Predicting...')
    predictions = np.argmax(d.model.predict(stack_x_test),axis=1)

    y_test = np.argmax(np.stack(y_test),axis=1)
    
    cf_matrix = confusion_matrix(y_test, predictions)
    acc    = accuracy_score(y_test, predictions)
    prec   = recall_score(y_test, predictions, average='macro')
    recall = precision_score(y_test, predictions, average='macro')
    f1     = f1_score(y_test, predictions, average='macro')
    print(cf_matrix)
    print('Accuracy: ', acc)
    print('Recall:   ', prec)
    print('Precision:', recall)
    print('F1:       ', f1)
    report.add(acc, prec, recall, f1, cf_matrix)

curr_time = current_time()
print('[Status] Ended at', curr_time)

report.export_to_file(''.join(['DIS-classification-report-', curr_time, '.txt']))

