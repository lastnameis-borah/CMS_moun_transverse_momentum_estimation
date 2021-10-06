import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense

def FCNN(dataset, predict):
    
    assert dataset in ('prompt_new', 'displaced', 'prompt_old')
    assert predict in ('pT', '1/pT', 'pT_classes')
    
    if dataset in ['prompt_new', 'displaced']:
        I = Input(shape=(16))
    if dataset == 'prompt_old':
        I = Input(shape=(12))
    x = Dense(512,activation='relu')(I)
    x = Dense(256,activation='relu')(x)
    x = Dense(128,activation='relu')(x)
    x = Dense(128,activation='relu')(x)
    x = Dense(128,activation='relu')(x)
    if predict=='pT':
        O = Dense(1,activation='linear')(x)
    if predict=='1/pT':
        O = Dense(1,activation='sigmoid')(x)
    if predict=='pT_classes':
        O = Dense(4,activation='softmax')(x)
    
    model = Model(inputs=I, outputs=O)
    
    return model
