import argparse, os, torch
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from Models.CNN import CNN
from Models.GNN import GNN
from Models.FCNN import FCNN
from Utils.PreprocessedData import get_data
from Utils.TFTrainer import train_nn
from Utils.TorchTrainer import train_gnn
from Utils.Plots import save_all_plots, plot_epochs_vs_loss, plot_ROC_AUC
from Utils.utility import *

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='Path of the csv file')
parser.add_argument('--dataset', type=str, default='prompt_new', help='select from (\'prompt_new\', \'displaced\', \'prompt_old\')')
parser.add_argument('--predict', type=str, default='pT', help='select from (\'pT\', \'1/pT\', \'pT_classes\')')
parser.add_argument('--model', type=str, default='FCNN', help = 'select from (\'FCNN\', \'CNN\', \'GNN\')')
parser.add_argument('--epochs', type=int, default=50, help='# epochs to train the model')
parser.add_argument('--batch_size', type=int, default=512, help='batch size used for training')
parser.add_argument('--folds', type=str, default="0,1,2,3,4,5,6,7,8,9", help='Folds to run the models')
parser.add_argument('--results', type=str, default="~/results/", help='Path of the folder to save results')
config = parser.parse_args()

path = config.path
dataset = config.dataset
predict = config.predict
model_name = config.model
epochs = config.epochs
batch_size = config.batch_size
folds = set(map(int,config.folds.strip().split(',')))
results_path = config.results

try:
    os.mkdir(results_path)
except:
    results_path = results_path

assert dataset in ('prompt_new', 'displaced', 'prompt_old')
assert predict in ('pT', '1/pT', 'pT_classes')
assert model_name in ('FCNN', 'CNN', 'GNN')

for fold, (X_train, Y_train, X_test, Y_test) in enumerate(get_data(path, dataset, predict, model_name)):
    if fold in folds:
        if model_name=='FCNN':
            model = FCNN(dataset, predict)
        if model_name=='CNN':
            model = CNN(dataset, predict)
        if model_name=='GNN':
            model = GNN(dataset, predict)
        if model_name in ('FCNN', 'CNN'):
            history = train_nn(model, predict, X_train, Y_train, X_test, Y_test, fold, epochs, batch_size, results_path, model_name)
            train_loss, val_loss = history.history['loss'], history.history['val_loss']
        else:
            train_loss, val_loss = train_gnn(model, predict, X_train, Y_train, X_test, Y_test, fold, epochs, batch_size, results_path)
        plot_epochs_vs_loss(train_loss, val_loss, results_path, fold)

df = merge_oofs(results_path, predict)
if predict=='pT_classes':
    ROC_AUC(df)
    plot_ROC_AUC(df, results_path)
else:
    save_all_plots(df, results_path)