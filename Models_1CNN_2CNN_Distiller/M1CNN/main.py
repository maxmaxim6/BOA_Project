

from utils import KFoldClassificationReport, current_time, load_data, preprocessing_m1cnn_classification
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import os

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN) 
from model import M1CNN
from numpy.random import seed
seed(7)
tf.random.set_seed(7)


'''''''''''''''''''''''''''
---------------------------
---- Main Starts here. ----
---------------------------
'''''''''''''''''''''''''''

print('[Status] Starting M1CNN...', current_time())
report = KFoldClassificationReport()
# Read data
print('[Status] Reading data...')

input= r'C:\Users\maxma\Desktop\PROJECT\mapG.csv'

#  run the algorithm and load the data into 
for x_train, y_train, x_test, y_test in load_data(input, preprocessing_m1cnn_classification, 'label', ['udps.n_bytes', 'label']):
    x_train = np.stack(x_train['udps.n_bytes'].values)
    y_train = np.stack(y_train)
    x_test = np.stack(x_test['udps.n_bytes'].values)
    y_test = np.stack(y_test)
    # Training


    classifier = M1CNN(payload_size=784, n_classes=6) 
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
# print('[Status] Ended at', curr_time)
# report.print()
report.export_to_file(''.join(['m1cnn-classification-report-', curr_time, '.txt']))