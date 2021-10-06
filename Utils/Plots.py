import os
import numpy as np
import pandas as pd
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc
from sklearn.metrics import mean_absolute_error as mae

def plot_epochs_vs_loss(train_loss, val_loss, results_path, fold):
    plt.figure()
    plt.plot(range(1,1+len(train_loss)), train_loss, label = 'TrainLoss')
    plt.plot(range(1,1+len(val_loss)), val_loss, label = 'ValLoss')
    plt.xlabel('Epoch-->')
    plt.ylabel('Loss-->')
    plt.title('Epochs vs Loss')
    plt.legend()
    plt.savefig(os.path.join(results_path, str(fold)+'_epochs_vs_loss.png'))
    plt.show()
    
def plot_mae(df, results_path ):
    dx = 0.5
    r = 100
    MAE1 = []
    for i in range(int(2/dx),int(150/dx)):
        P = df[(df['True_pT']>=(i-1)*dx)&(df['True_pT']<=(i+1)*dx)]
        p = mae(P['True_pT'],P['Predicted_pT'])
        if p<100:
            p=p
        else:
            p=p_
        MAE1.append(p)
        p_=p
    MAE1 = [0]*2*int(1/dx)+MAE1[:r*2-2*int(1/dx)]
    plt.figure()
    plt.plot([i*dx for i in range(int(r/dx))],MAE1)
    plt.xlabel('pT (in GeV) -->')
    plt.ylabel('MAE -->')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'mae.png'))
    plt.show()
    return MAE1

def plot_mae_pT(df, results_path ):
    dx = 0.5
    r = 100
    MAE1 = []
    for i in range(int(2/dx),int(150/dx)):
        P = df[(df['True_pT']>=(i-1)*dx)&(df['True_pT']<=(i+1)*dx)]
        p = mae(P['True_pT'],P['Predicted_pT'])/(i)
        if p<100:
            p=p
        else:
            p=p_
        MAE1.append(p)
        p_=p
    MAE1 = [0]*2*int(1/dx)+MAE1[:r*2-2*int(1/dx)]
    plt.figure()
    plt.plot([i*dx for i in range(int(r/dx))],MAE1)
    plt.xlabel('pT (in GeV) -->')
    plt.ylabel('MAE/pT -->')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'mae_pt.png'))
    plt.show()
    return MAE1
    
def plot_f1_pT_upper(df, results_path ):
    f1 = []
    for i in range(5,121):
        f1.append(f1_score(df['True_pT']>=i, df['Predicted_pT']>=i))
    plt.figure()
    plt.plot(range(5,121),f1)
    plt.xlabel('pT (in GeV) -->')
    plt.ylabel('F1 (for class pT < x) -->')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'f1(pT>x).png'))
    plt.show()
    return f1
    
def plot_f1_pT_lower(df, results_path ):
    f1 = []
    for i in range(5,121):
        f1.append(f1_score(df['True_pT']<=i, df['Predicted_pT']<=i))
    plt.figure()
    plt.plot(range(5,121),f1)
    plt.xlabel('pT (in GeV) -->')
    plt.ylabel('F1 (for class pT < x) -->')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'f1(pT<x).png'))
    plt.show()
    return f1
    
    
def plot_accuracy(df, results_path ):
    acc = []
    for i in range(5,121):
        acc.append(accuracy_score(df['True_pT']>=i, df['Predicted_pT']>=i))
    plt.figure()
    plt.plot(range(5,121),acc)
    plt.xlabel('pT (in GeV) -->')
    plt.ylabel('Accuracy -->')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'acc.png'))
    plt.show()
    return acc

def save_all_plots(df, results_path ):
    plot_accuracy(df, results_path )
    plot_f1_pT_lower(df, results_path )
    plot_f1_pT_upper(df, results_path )
    plot_mae_pT(df, results_path )
    plot_mae(df, results_path )
    
def plot_ROC_AUC(df, results_path):
    df['pT_classes']=df['true_value'].to_list()
    classes = ['0-10','10-30','30-100','100-inf','micro','macro']
    for i in range(6):
        if i<4:
            fpr,tpr,_ = roc_curve(df['pT_classes']==i, df[classes[i]])
            roc_auc = auc(fpr, tpr) 
        if i==4:
            y_score = df[classes[:4]].to_numpy()
            y_test = np.array([df['pT_classes']==0,df['pT_classes']==1,df['pT_classes']==2,df['pT_classes']==3]).T*1.0
            fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())
            roc_auc = auc(fpr, tpr)
        if i==5:
            y_score = df[classes[:4]].to_numpy()
            y_test = np.array([df['pT_classes']==0,df['pT_classes']==1,df['pT_classes']==2,df['pT_classes']==3]).T*1.0
            all_fpr = np.unique(np.concatenate([roc_curve(df['pT_classes']==i, df[classes[i]])[0] for i in range(4)]))
            mean_tpr = np.zeros_like(all_fpr)
            for j in range(4):
                A = roc_curve(df['pT_classes']==j, df[classes[j]])
                mean_tpr += interp(all_fpr, A[0], A[1])
            mean_tpr /= 4
            fpr = all_fpr
            tpr = mean_tpr
            roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic ( '+classes[i]+ ' )')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(results_path, 'roc_curve_'+classes[i]+'.png'))
        plt.show()