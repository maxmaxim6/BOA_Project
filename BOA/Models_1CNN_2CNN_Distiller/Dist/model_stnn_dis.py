import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.constraints import max_norm
import numpy as np

#  model and getters 

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

def get_sm_layers_payload_modality(input_layer):
    return stack([
        input_layer,
        BatchNormalization(),
        Conv1D(16, 25, name='Conv1D_payload_1'),
        ReLU(),
        MaxPooling1D(3, name='MaxPooling1D_payload_1'),
        Conv1D(32, 35, name='Conv1D_payload_2'),
        ReLU(),
        MaxPooling1D(3, name='MaxPooling1D_payload_2'),
        Flatten(), # added because of the mail
        Dropout(0.2),
        Dense(128),
        ReLU()
    ])

def get_sm_layers_protocol_fields_modality(input_layer):
    # SM = Single Modality
    return stack([
        input_layer,
        BatchNormalization(),
        Bidirectional(GRU(64, return_sequences=True, kernel_constraint=max_norm(3))),
        ReLU(),
        Flatten(), # added because of the mail
        Dropout(0.2),
        Dense(128),
        ReLU()
    ])
def get_sm_layers_stnn_modality(input_layer):
    # SM = Single Modality
    return stack([
        input_layer,
        BatchNormalization(),
        Bidirectional(LSTM(65,return_sequences=True)),
        Lambda(lambda x: tf.expand_dims(x, axis=3)),
        Conv2D(32,3,padding='same'),
        LeakyReLU(),
        Conv2D(32,3,padding='same'),
        LeakyReLU(),
        MaxPool2D(2),
        Conv2D(64,3,padding='same'),
        LeakyReLU(),
        Conv2D(128,3,padding='same'),
        LeakyReLU(),
        MaxPool2D(2),
        Flatten(),
        Dense(512),
        Dropout(0.2),
        Dense(128),
        ReLU()
    ])
    
def get_sr_layers():
    # SR = Shared Representation
    return [
        Dropout(0.2),
        Dense(128),
        ReLU(),
        Dropout(0.2),
    ]
    
def get_ts_layers(classes_count):
    # TS = Task Specific
    return [
        Dense(128),
        ReLU(),
        Dropout(0.2),
        Dense(classes_count),
        Softmax()
    ]

class Distiller:


    def __init__(self, n_classes) -> None:
        self.n_classes = n_classes

        input_layer_payload_modality         = Input(shape=(784,1), name='input_payload')
        input_layer_protocol_fields_modality = Input(shape=(32,4), name='input_protocol_fields')
        input_layer_stnn_modality = Input(shape=(5,14), name='input_stnn')

        merged_modal_layers = Concatenate()([
            get_sm_layers_payload_modality(input_layer_payload_modality),
            get_sm_layers_protocol_fields_modality(input_layer_protocol_fields_modality),
            get_sm_layers_stnn_modality(input_layer_stnn_modality)
        ])
        sr_layers = [merged_modal_layers] + get_sr_layers()
        
        self.payload_model = Model(
            name='Distiller',
            inputs=input_layer_payload_modality,
            outputs=get_sm_layers_payload_modality(input_layer_payload_modality)
        )
        
        self.proto_model = Model(
            name='Distiller',
            inputs=input_layer_protocol_fields_modality,
            outputs=get_sm_layers_protocol_fields_modality(input_layer_protocol_fields_modality)
        )

        self.stnn_model = Model(
            name='Distiller',
            inputs=input_layer_stnn_modality,
            
            outputs=get_sm_layers_stnn_modality(input_layer_stnn_modality)
        )
        

        shared_representation = stack(
            [
                Concatenate()([
                    self.payload_model.output,
                    self.proto_model.output,
                    self.stnn_model.output
                ])
            ]
            + get_sr_layers()
        )
        
        # A
        self.model = Model(
            name='Distiller',
            inputs=[
                self.payload_model.input,
                self.proto_model.input,
                self.stnn_model.input,
            ],
            outputs=[

                stack([shared_representation] + get_ts_layers(classes_count=n_classes))
            ]
        )


        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        self.model.compile(
            optimizer='adam',
            loss=loss_fn,
            metrics=['accuracy']
        )

    def get_model_for_pretraining(self, model):

        return Model(
            name='Distiller',
            inputs=model.input,
            outputs=[

                    stack([model.output, Dense(self.n_classes, activation='softmax')])

            ]
        )

    def freeze_for_finetuning(self):
        for layer in self.payload_model.layers[0:8]:
            layer.trainable = False

        for layer in self.proto_model.layers[0:4]:
            layer.trainable = False
        for layer in self.stnn_model.layers[0:14]:
            layer.trainable = False
        
    def unfreeze_for_finetuning(self):
        for layer in self.payload_model.layers[0:8]:
            layer.trainable = True

        for layer in self.proto_model.layers[0:4]:
            layer.trainable = True

        for layer in self.stnn_model.layers[0:14]:
            layer.trainable = True
        
