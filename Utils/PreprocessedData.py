import gc, random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def get_data(path, dataset, predict, model_name, fold=[True]*10):
    
    assert dataset in ('prompt_new', 'displaced', 'prompt_old')
    assert predict in ('pT', '1/pT', 'pT_classes')
    
    df = pd.read_csv(path)
    df = df.drop(columns = [i for i in df.columns if '_1' in i])
    df['non_hits'] = df[[i for i in df.columns if 'mask' in i]].sum(axis=1)
    df = df[df['non_hits']==0].reset_index(drop=True)

    df['1/pT'] = df['q/pt'].abs()
    
    def label(a):
        if a<=10:
            return 0
        if a>10 and a<=30:
            return 1
        if a>30 and a<=100:
            return 2
        if a>100:
            return 3

    df['pT'] = 1/df['1/pT']

    df['pT_classes'] = df['pT'].apply(label)

    if dataset in ['prompt_new', 'displaced']:
        features = ['emtf_phi_'+str(i) for i in [0,2,3,4]] + ['emtf_theta_'+str(i) for i in [0,2,3,4]] + ['fr_'+str(i) for i in [0,2,3,4]] + ['old_emtf_phi_'+str(i) for i in [0,2,3,4]]
    if dataset=='prompt_old':
        features = ['Phi_'+str(i) for i in [0,2,3,4]] + ['Theta_'+str(i) for i in [0,2,3,4]] + ['Front_'+str(i) for i in [0,2,3,4]]
    if predict=='1/pT':
        label = ['1/pT']
    if predict=='pT':
        label = ['pT']
    if predict=='pT_classes':
        label = ['pT_classes']

    scaler_1 = StandardScaler()
    df[features] = scaler_1.fit_transform(df[features])
    
    shuffled_list = list(range(len(df)))
    random.Random(242).shuffle(shuffled_list)
    shuffled_list = np.array_split(np.array(shuffled_list), 10)
    
    for i in range(10):
        if fold[i]:
            X_train = df[features].iloc[np.concatenate([shuffled_list[j] for j in range(10) if j !=i])]
            Y_train = df[label].astype('float32').iloc[np.concatenate([shuffled_list[j] for j in range(10) if j!=i])]

            X_test = df[features].iloc[shuffled_list[i]]
            Y_test = df[label].astype('float32').iloc[shuffled_list[i]]
            
            if model_name=='CNN' and dataset=='prompt_old':
                X_train['feat0'] = 0
                X_train['feat1'] = 0
                X_train['feat2'] = 0
                X_train['feat3'] = 0
                X_test['feat0'] = 0
                X_test['feat1'] = 0
                X_test['feat2'] = 0
                X_test['feat3'] = 0
            
            yield X_train, Y_train, X_test, Y_test
            
            del X_train, Y_train, X_test, Y_test
            gc.collect()