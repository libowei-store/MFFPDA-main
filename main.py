import argparse
import csv

import fitlog
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

import pickle
import os

from utils import *
from sklearn import metrics
from tqdm import tqdm
from torch.utils.data import DataLoader

import random
from sklearn.model_selection import train_test_split, StratifiedKFold
from network import MFFPDA
from source.similarity import *

import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def train(model, loader, criterion, opt, device):
    model.train()
    criterion2 = torch.nn.MSELoss()
    for idx, data in enumerate(tqdm(loader, desc='Iteration')):  # tqdm是进度条  返回 enumerate(枚举) 对象。


        probiotic, disease, label = data

        output = model(probiotic, disease, device)
        loss = criterion(output, label.float().to(device))

        opt.zero_grad()
        loss.backward()
        opt.step()

    print('Train Loss:{}'.format(loss))
    return loss


def validate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in tqdm(loader, desc='Iteration'):

            probiotic, disease, label = data
            output = model(probiotic, disease, device)

            y_true.append(label)
            y_pred.append(output)

    y_true = torch.cat(y_true, dim=0).cpu()
    y_pred = torch.cat(y_pred, dim=0).cpu()
    for i in range(len(y_pred2)):
        if y_pred2[i] >= 0.5:
            y_pred2[i] = 1
        else:
            y_pred2[i] = 0
    iprecision, irecall, ithresholds = metrics.precision_recall_curve(y_true,
                                                                      y_pred,
                                                                      pos_label=1,
                                                                      sample_weight=None)
    aupr = metrics.auc(irecall, iprecision)
    auc = metrics.roc_auc_score(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred2)
    recall = metrics.recall_score(y_true, y_pred2, average='macro')
    precision = metrics.precision_score(y_true, y_pred2, average='macro')
    f1 = metrics.f1_score(y_true, y_pred2, average='macro')

    return aupr, auc, acc, f1, recall, precision


def setup_seed(seed):
    torch.manual_seed(seed)  #
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  #
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_data(args):
    rawdata_dir = args.data_path
    with open(args.data_path + 'association_matrix.pkl', 'rb') as f:
        final_sample = pickle.load(f)
    final_positive_sample, final_negative_sample = Extract_positive_negative_samples(
        final_sample, addition_negative_number='all')
    final_sample = np.vstack((final_positive_sample, final_negative_sample))
    X = final_sample[:, 0::]

    data = []
    data_x = []
    data_y = []

    for i in range(X.shape[0]):
        data_x.append((X[i, 0], X[i, 1]))
        data_y.append((int(float(X[i, 2]))))
        data.append((X[i, 0], X[i, 1], X[i, 2]))
    fold = 1
    kfold = StratifiedKFold(10, random_state=40, shuffle=True)
    total_auc, total_pr_auc, total_acc, total_f1, total_recall, total_precision = [], [], [], [], [], []

    for k, (Train, val_test_set) in enumerate(kfold.split(data_x, data_y)):
        print("==================================fold {} start".format(fold))
        data = np.array(data)
        train_set = data[Train].tolist()
        val_test_data = data[val_test_set].tolist()

        val_set, test_set = train_test_split(val_test_set, test_size=0.5, random_state=40)
        val_set = data[val_set].tolist()
        test_set = data[test_set].tolist()

        test_set = np.array(test_set)

        probiotic_features, disease_features = read_raw_data(rawdata_dir, val_test_data)

        probiotic_features_matrix = probiotic_features[0]
        for i in range(1, len(probiotic_features)):
            probiotic_features_matrix = np.hstack((probiotic_features_matrix, probiotic_features[i]))

        disease_features_matrix = disease_features[0]
        for i in range(1, len(disease_features)):
            disease_features_matrix = np.hstack((disease_features_matrix, disease_features[i]))

        Dataset = MyDataset
        train_dataset = Dataset(probiotic_features_matrix, disease_features_matrix, train_set)
        test_dataset = Dataset(probiotic_features_matrix, disease_features_matrix, test_set)
        val_dataset = Dataset(probiotic_features_matrix, disease_features_matrix, val_set)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)


        model = MFFPDA(72*5, 235*6, args.embed_dim, args.batch_size, args.droprate, args.droprate).to(args.device)
        if args.mode == 'train':
            Regression_criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            log_folder = os.path.join(os.getcwd(), "result/log_gene_best2", model._get_name())
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)
            fitlog.set_log_dir(log_folder)
            fitlog.add_hyper(args)
            fitlog.add_hyper_in_file(__file__)

            stopper = EarlyStopping(mode='higher', patience=args.patience)
            for epoch in range(1, args.epochs + 1):

                print("=====Epoch {}".format(epoch))
                print("Training...")
                train_loss = train(model, train_loader, Regression_criterion, optimizer, args.device)

                fitlog.add_loss(train_loss.item(), name='Train AUC', step=epoch)

                print('Evaluating...')
                aupr_v, auc_v, acc_v, f1_v, recall_v, precision_v = validate(model, val_loader, args.device)


                print("Validation aupr:{} auc:{} acc:{} f1:{} recall:{} precision:{}".format(aupr_v, auc_v, acc_v, f1_v, recall_v, precision_v))
                fitlog.add_metric({'val': {'AUPR': aupr_v}}, step=epoch)

                early_stop = stopper.step(auc_v, f1_v, model)
                if early_stop:
                    break

            print('EarlyStopping! Finish training!')
            print('Testing...')
            stopper.load_checkpoint(model)
            if not os.path.exists('/weight'):
                os.makedirs('/weight')
            torch.save(model.state_dict(), 'weight/fold_{}_{}.pth'.format(fold, args.weight_path))
            train_aupr, train_auc, train_acc, train_f1, train_recall, train_precision = validate(model, train_loader, args.device)
            val_aupr, val_auc, val_acc, val_f1, val_recall, val_precision = validate(model, val_loader, args.device)
            test_aupr, test_auc, test_acc, test_f1, test_recall, test_precision = validate(model, test_loader, args.device)
            print('Train reslut: aupr:{} auc:{} acc:{} f1:{} recall:{} precision:{}'.format(train_aupr, train_auc,  train_acc, train_f1, train_recall, train_precision))
            print('Val reslut: aupr:{} auc:{} acc:{} f1:{} recall:{} precision:{}'.format(val_aupr, val_auc, val_acc, val_f1, val_recall, val_precision))
            print('Test reslut: aupr:{} auc:{} acc:{} f1:{} recall:{} precision:{}'.format(test_aupr, test_auc, test_acc, test_f1, test_recall, test_precision))
            total_auc.append(test_auc)
            total_pr_auc.append(test_aupr)
            total_acc.append(test_acc)
            total_f1.append(test_f1)
            total_recall.append(test_recall)
            total_precision.append(test_precision)
            fitlog.add_best_metric(
                {'epoch': epoch - args.patience,
                 "train": {'AUPR': train_aupr, 'auc': train_auc, 'acc': train_acc, 'f1': train_f1, 'recall': train_recall, 'precision': train_precision},
                 "valid": {'AUPR': val_aupr, 'auc': val_auc, 'acc': val_acc, 'f1': val_f1, 'recall': val_recall, 'precision': val_precision},
                 "test": {'AUPR': test_aupr, 'auc': test_auc, 'acc': test_acc, 'f1': test_f1, 'recall': test_recall, 'precision': test_precision}})

        elif args.mode == 'test':

            model.load_state_dict(
                torch.load('weight/fold_{}_{}.pth'.format(fold, args.weight_path), map_location=args.device)['model_state_dict'])
            test_aupr, test_auc, test_acc, test_f1, test_recall, test_precision = validate(model, test_loader, args.device)
            print('Test AUPR: {}, AUC: {}, ACC: {}, F1: {}, RECALL: {}, PRECISION: {}'.format(round(test_aupr, 4), round(test_auc, 4), round(test_acc, 4), round(test_f1, 4), round(test_recall, 4), round(test_precision, 4)))
        print("==================================fold {} end".format(fold))
        fold += 1
    print('Total_AUC:')
    print(str(np.around(np.mean(total_auc), 4)) + '±' + str(np.around(np.std(total_auc), 4)))
    print('Total_AUPR:')
    print(str(np.around(np.mean(total_pr_auc), 4)) + '±' + str(np.around(np.std(total_pr_auc), 4)))
    print('Total_ACC:')
    print(str(np.around(np.mean(total_acc), 4)) + '±' + str(np.around(np.std(total_acc), 4)))
    print('Total_F1:')
    print(str(np.around(np.mean(total_f1), 4)) + '±' + str(np.around(np.std(total_f1), 4)))
    print('Total_RECALL:')
    print(str(np.around(np.mean(total_recall), 4)) + '±' + str(np.around(np.std(total_recall), 4)))
    print('Total_PRECISION:')
    print(str(np.around(np.mean(total_precision), 4)) + '±' + str(np.around(np.std(total_precision), 4)))



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Model')
    parser.add_argument('--epochs', type=int, default=128,
                        metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001,
                        metavar='FLOAT', help='learning rate')
    parser.add_argument('--embed_dim', type=int, default=128,
                        metavar='N', help='embedding dimension')
    parser.add_argument('--weight_decay', type=float, default=0.0003,
                        metavar='FLOAT', help='weight decay')
    parser.add_argument('--droprate', type=float, default=0.5,
                        metavar='FLOAT', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        metavar='N', help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=64,
                        metavar='N', help='input batch size for testing')
    parser.add_argument('--data_path', type=str, default='data/',
                        metavar='STRING', help='data_path')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for earlystopping (default: 10)')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or test')
    parser.add_argument('--weight_path', type=str, default='best2',
                        help='filepath for pretrained weights')
    args = parser.parse_args()

    print('-------------------- Hyperparams --------------------')
    print('weight decay: ' + str(args.weight_decay))
    print('dropout rate: ' + str(args.droprate))
    print('learning rate: ' + str(args.lr))
    print('dimension of embedding: ' + str(args.embed_dim))
    # setup_seed(60)
    load_data(args)


if __name__ == "__main__":
    main()
    print('Finish!!!')
