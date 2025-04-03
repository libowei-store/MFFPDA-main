import pickle
import numpy as np
import torch

from source.similarity import *
from source.dual_transformation_networks import dual_transformation_networks
from source.NMF import NMF_calculate
from source.probiotic_function_similarity import probiotic_function_similarity
import random
import datetime
import os

from torch.utils.data import Dataset

def read_raw_data(rawdata_dir, val_test_set):
    gii = open(rawdata_dir + '/' + 'probiotic_squence_feature_matrix.pkl', 'rb')
    probiotic_squence_similarity = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + '/' + 'disease_function_similarity_matrix.pkl', 'rb')
    disease_function_similarity = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + '/' + 'disease_semantic_similarity_matrix1.pkl', 'rb')
    disease_semantic_similarity1 = pickle.load(gii)
    disease_semantic_similarity1 = np.float32(disease_semantic_similarity1)
    gii.close()

    gii = open(rawdata_dir + '/' + 'disease_semantic_similarity_matrix2.pkl', 'rb')
    disease_semantic_similarity2 = pickle.load(gii)
    disease_semantic_similarity2 = np.float32(disease_semantic_similarity2)
    gii.close()

    gii = open(rawdata_dir + '/' + 'disease_symptom_similarity_matrix.pkl', 'rb')
    disease_symptom_similarity = pickle.load(gii)
    gii.close()

    ggi = open(rawdata_dir + '/' + 'association_matrix.pkl', 'rb')
    assoction_matrix = pickle.load(ggi)
    gii.close()

    for i in range(len(val_test_set)):
        assoction_matrix[val_test_set[i][0]][val_test_set[i][1]] = 0

    probiotic_HIP_similarity = HIP_Calculate(assoction_matrix)

    disease_HIP_similarity = HIP_Calculate(assoction_matrix.T)


    probiotic_function_similarity1 = probiotic_function_similarity(assoction_matrix, disease_semantic_similarity1)
    probiotic_function_similarity2 = probiotic_function_similarity(assoction_matrix, disease_semantic_similarity2)

    probiotic_NFM_matrix = NMF_calculate(assoction_matrix.T, 25)
    disease_NFM_matrix = NMF_calculate(assoction_matrix, 50)
    probiotic_NFM_matrix = Cosine_similarity(probiotic_NFM_matrix)
    disease_NFM_matrix = Cosine_similarity(disease_NFM_matrix)

    probiotic_feature, disease_feature = [], []

    probiotic_feature.append(probiotic_squence_similarity)
    probiotic_feature.append(probiotic_HIP_similarity)
    probiotic_feature.append(probiotic_function_similarity1)
    probiotic_feature.append(probiotic_function_similarity2)
    probiotic_feature.append(probiotic_NFM_matrix)

    disease_feature.append(disease_symptom_similarity)
    disease_feature.append(disease_function_similarity)
    disease_feature.append(disease_semantic_similarity1)
    disease_feature.append(disease_semantic_similarity2)
    disease_feature.append(disease_HIP_similarity)
    disease_feature.append(disease_NFM_matrix)

    return probiotic_feature, disease_feature


def Extract_positive_negative_samples(DAL, addition_negative_number='all'):
    k = 0
    interaction_target = np.zeros((DAL.shape[0] * DAL.shape[1], 3)).astype(int)
    for i in range(DAL.shape[0]):
        for j in range(DAL.shape[1]):
            interaction_target[k, 0] = i
            interaction_target[k, 1] = j
            interaction_target[k, 2] = DAL[i, j]
            k = k + 1
    data_shuffle = interaction_target[interaction_target[:, 2].argsort()]
    number_positive = len(np.nonzero(data_shuffle[:, 2])[0])
    final_positive_sample = data_shuffle[interaction_target.shape[0] - number_positive::]
    negative_sample = data_shuffle[0:interaction_target.shape[0] - number_positive]
    a = np.arange(interaction_target.shape[0] - number_positive)
    a = list(a)
    if addition_negative_number == 'all':
        b = random.sample(a, (interaction_target.shape[0] - number_positive))
    else:
        b = random.sample(a, (1 + addition_negative_number) * number_positive)
    final_negtive_sample = negative_sample[b[0:number_positive], :]
    addition_negative_sample = negative_sample[b[number_positive::], :]
    print(final_negtive_sample.shape)
    print(addition_negative_sample.shape)
    return final_positive_sample, final_negtive_sample


class EarlyStopping():
    """
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
        If ``metric`` is not None, then mode will be determined
        automatically from that.
    patience : int
        The early stopping will happen if we do not observe performance
        improvement for ``patience`` consecutive epochs.
    filename : str or None
        Filename for storing the model checkpoint. If not specified,
        we will automatically generate a file starting with ``early_stop``
        based on the current time.
    """

    def __init__(self, mode='higher', patience=10, filename=None, metric=None):
        if filename is None:
            dt = datetime.datetime.now()
            folder = os.path.join(os.getcwd(), 'results')
            if not os.path.exists(folder):
                os.makedirs(folder)
            filename = os.path.join(folder, 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
                dt.date(), dt.hour, dt.minute, dt.second))

        if metric is not None:
            assert metric in ['r2', 'mae', 'rmse', 'auc', 'aupr', 'acc', 'f1', 'recall', 'precision', 'loss'], \
                "Expect metric to be 'r2' or 'mae' or " \
                "'rmse' or 'auc', got {}".format(metric)
            if metric in ['r2', 'auc', 'aupr', 'acc', 'f1', 'recall', 'precision']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse', 'loss']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score1 = None
        self.best_score2 = None
        self.early_stop = False

    def _check_higher(self, score1, prev_best_score1, score2, prev_best_score2):
        """Check if the new score is higher than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is higher than the previous best score.
        """
        return score1 > prev_best_score1 and score2 > prev_best_score2

    def _check_lower(self, score1, prev_best_score1, score2, prev_best_score2):
        """Check if the new score is lower than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is lower than the previous best score.
        """
        return score1 < prev_best_score1 and score2 < prev_best_score2

    def step(self, score1, score2, model):
        """Update based on a new score.
        The new score is typically model performance on the validation set
        for a new epoch.
        Parameters
        ----------
        score : float
            New score.
        model : nn.Module
            Model instance.
        Returns
        -------
        bool
            Whether an early stop should be performed.
        """
        if self.best_score1 is None and self.best_score2 is None:
            self.best_score1 = score1
            self.best_score2 = score2
            self.save_checkpoint(model)
        elif self._check(score1, self.best_score1, score2, self.best_score2):
            self.best_score1 = score1
            self.best_score2 = score2
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)

    def load_checkpoint(self, model):
        '''Load the latest checkpoint
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])

class MyDataset(Dataset):
    def __init__(self, probiotic_dict, disease_dict, IC):
        super(MyDataset, self).__init__()
        self.probiotic, self.disease = probiotic_dict, disease_dict
        IC = np.array(IC)
        self.probiotic_name = IC[:, 0]
        self.disease_name = IC[:, 1]
        self.value = IC[:, 2]

    def __len__(self):
        return len(self.value)

    def __getitem__(self, index):
        return (self.probiotic[int(self.probiotic_name[index])], self.disease[int(self.disease_name[index])],
                self.value[index]
                )
