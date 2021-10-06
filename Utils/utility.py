import os
import pandas as pd
from sklearn.metrics import roc_auc_score

def merge_oofs(results_path, predict):
    
    files = [ i for i in os.listdir(results_path) if i[:10]=='OOF_preds_']
    df = pd.concat([pd.read_csv(os.path.join(results_path,file)) for file in files], axis = 0).reset_index(drop=True)
    
    if predict=='pT':
        df['True_pT'] = df['true_value']
        df['Predicted_pT'] = df['preds']
    if predict=='1/pT':
        df['True_pT'] = 1/df['true_value']
        df['Predicted_pT'] = 1/df['preds']
    
    df.to_csv(os.path.join(results_path,'OOF_preds.csv'), index=False)
    return df
        
def ROC_AUC(df):
    print('ROC-AUC - 0-10 GeV - ', roc_auc_score((df['true_value']==0)*1.0, df['0-10']))
    print('ROC-AUC - 10-30 GeV - ', roc_auc_score((df['true_value']==1)*1.0, df['10-30']))
    print('ROC-AUC - 30-100 GeV - ', roc_auc_score((df['true_value']==2)*1.0, df['30-100']))
    print('ROC-AUC - 100-inf GeV - ', roc_auc_score((df['true_value']==3)*1.0, df['100-inf']))
    return None