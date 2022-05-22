

from utils import KFoldClassificationReport, current_time, load_data, preprocessing_m2cnn_classification, preprocessing_m2cnn_detection
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)  
from model import M2CNN
from numpy.random import seed
import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import time
seed(7)
tf.random.set_seed(7)


'''''''''''''''''''''''''''
---------------------------
---- Main Starts here. ----
---------------------------
'''''''''''''''''''''''''''

print('[Status] Starting M2CNN...', current_time())
report = KFoldClassificationReport()
# Read data
print('[Status] Reading data...')

path= r'C:\Users\maxma\Desktop\PROJECT\mapG.csv'

for x_train, y_train, x_test, y_test in load_data(path, preprocessing_m2cnn_classification,'label', ['udps.n_bytes', 'label']):
    x_train = np.stack(x_train['udps.n_bytes'].values)
    y_train = np.stack(y_train)
    x_test = np.stack(x_test['udps.n_bytes'].values)
    y_test = np.stack(y_test)
    # Training
    print('[Status] Training model...')

    print('Amount of instances', len(y_train))

    classifier = M2CNN(payload_size=784, n_classes=6) 
    x_train
    print(classifier.model.summary())
    epochs = 35
    batch_size = 48
    classifier.model.fit(
        x_train, 
        y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.005, patience=5)],
        use_multiprocessing=True,
        workers=2,
        verbose=2
    )

    print('[Status] Predicting...')
    predictions = classifier.model.predict(x_test)
    predictions = np.argmax(predictions, axis=1)
    y_test = np.argmax(y_test, axis=1)
    
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
# report.print()
# print(report.cf_matrices)
report.export_to_file(''.join(['m2cnn-classification-report-', curr_time, '.txt']))