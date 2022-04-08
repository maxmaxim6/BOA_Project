import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow.keras as K
from utils import stack


class M1CNN:
    def __init__(self, payload_size=784, n_classes=2) -> None:
        input_layer = Input(shape=(payload_size,1))
        self.model = Model(
            name='M1CNN',
            inputs=input_layer,
            outputs=stack([
                input_layer, # first layer
                Conv1D(32, 25, strides=1, padding="same", activation='relu'),
                MaxPooling1D(3, strides=3, padding="same",),
                Conv1D(64, 25, strides=1, padding="same", activation='relu'),
                MaxPooling1D(3, strides=3, padding="same",),
                Flatten(),
                Dense(1024, activation='relu'),
                Dropout(0.2),
                Dense(n_classes, activation='softmax'),
            ])
        )
        
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.Precision()
            ]
        )