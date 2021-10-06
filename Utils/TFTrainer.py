import pandas as pd, os
import tensorflow as tf
import tensorflow.keras.backend as K
from focal_loss import SparseCategoricalFocalLoss
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def pTLossTF(y_true,y_pred):
    y_t = K.cast(y_true<80,K.dtype(y_true))*y_true + K.cast(y_true>=80,K.dtype(y_true))*K.cast(y_true<250,K.dtype(y_true))*y_true*2.4 + K.cast(y_true>=160,K.dtype(y_true))*10 
    return K.mean(y_t*K.pow((y_pred-y_true)/y_true,2))/250

def train_nn(model, predict, X_train, Y_train, X_test, Y_test, fold=0, epochs=50, batch_size=512, results_path='./', model_name='FCNN'):
    
    assert predict in ('pT', '1/pT', 'pT_classes')
    
    test_index = list(X_test.index)
    X_train = X_train.reset_index(drop=True)
    Y_train = Y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)
    
    if model_name=='CNN':
        X_train = X_train.to_numpy().reshape((-1,4,4,1))
        X_test = X_test.to_numpy().reshape((-1,4,4,1))
    
    checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss',patience=3,verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=0,verbose=0)

    if predict=='1/pT':
        model.compile(optimizer = 'adam', loss='mse')
    if predict=='pT_classes':
        model.compile(optimizer = 'adam', loss=SparseCategoricalFocalLoss(gamma=2))
    if predict=='pT':
        model.compile(optimizer = 'adam', loss=pTLossTF)
        
    history = model.fit(x = X_train, y = Y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_split=0.1, callbacks=[checkpoint,early_stop,reduce_lr])
    
    model.load_weights("model.h5")
    
    P = model.predict(X_test)
    
    OOF_preds = pd.DataFrame()
    OOF_preds['row'] = test_index
    OOF_preds['true_value'] = Y_test[Y_test.columns[0]].to_list()
    if predict in ('pT', '1/pT'):
        OOF_preds['preds'] = P.reshape((len(X_test)))
    else:
        OOF_preds['0-10'] = P[:,0].reshape((len(X_test)))
        OOF_preds['10-30'] = P[:,1].reshape((len(X_test)))
        OOF_preds['30-100'] = P[:,2].reshape((len(X_test)))
        OOF_preds['100-inf'] = P[:,3].reshape((len(X_test)))
    OOF_preds.to_csv(os.path.join(results_path, 'OOF_preds_'+str(fold)+'.csv'), index=False)
    
    return history