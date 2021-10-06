import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten

def CNN(dataset, predict):
    
    assert dataset in ('prompt_new', 'displaced', 'prompt_old')
    assert predict in ('pT', '1/pT', 'pT_classes')
    
    activation = tf.keras.layers.LeakyReLU(0.15)
    I = Input(shape=(4,4,1))
    x = Conv2D(512, kernel_size=(2, 2),activation=activation)(I)
    x = Conv2D(128, kernel_size=(2, 2),activation=activation)(x)
    x = Conv2D(128, kernel_size=(2, 2),activation=activation)(x)
    x = Flatten()(x)
    x = Dense(256, activation=activation)(x)
    x = Dense(128, activation=activation)(x)
    x = Dense(128, activation=activation)(x)
    x = Dense(64, activation=activation)(x)
    if predict=='pT':
        O = Dense(1,activation='linear')(x)
    if predict=='1/pT':
        O = Dense(1,activation='sigmoid')(x)
    if predict=='pT_classes':
        O = Dense(4,activation='softmax')(x)

    model = Model(inputs=I, outputs=O)

    return model