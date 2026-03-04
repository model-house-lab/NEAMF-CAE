import re
import numpy as np
import os
import cv2
from model_upload import *
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, roc_curve, auc, precision_score, recall_score, roc_auc_score, accuracy_score
from tensorflow.keras.utils import Sequence


def sort_key_zidian_1(path):
    match = re.search(r'\d+', str(path))
    if match:
        return int(match.group(0))
    return float('inf')


def sort_key_zidian_2(file_name):
    symbol_part = re.findall(r'[+]+', file_name)
    symbol_count = len(symbol_part[0]) if symbol_part else 0

    num_part = re.findall(r'\d+', file_name)
    num_part = list(map(int, num_part))

    return symbol_count, num_part

def read_images_from_folders(folder_paths):
    images = []
    for path_pair in folder_paths:

        folder_images = []
        channel_images = []

        index = 0
        for filename in os.listdir(path_pair[0]):
            if filename.endswith(".bmp"):
                image_path = path_pair[0]+ "\\" + str(index) + ".bmp"
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                channel_images.append(image)
            index += 1
        folder_images.append(np.array(channel_images))

        channel_images = []
        index = 0
        for filename in os.listdir(path_pair[1]):
            if filename.endswith(".bmp"):
                image_path = os.path.join(path_pair[1], filename)
                image_path = path_pair[1] + "\\" + str(index) + ".bmp"
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                channel_images.append(image)
            index += 1
        folder_images.append(np.array(channel_images))

        images.append(np.stack(folder_images, axis=-1))
    return np.array(images)

def focal_region_loss(y_true, y_pred):
    X_true = y_true[0]
    mask_true = y_true[1]
    print("y_true.shape==", y_true.shape)
    print("X_true.shape==",X_true.shape,"y_pred.shape==",y_pred.shape,"mask_true.shape==",mask_true.shape)
    loss = K.mean(K.square(X_true - y_pred) * (1 + mask_true))
    return loss

def sort_key_32(path):
    path_str = str(path)
    digits = re.findall(r'\d+', path_str)
    if digits:
        return int(digits[1])
    else:
        return float('inf')

def compute_95ci_bootstrap(y_true, y_prob, y_pred, n_bootstraps=1000, rng_seed=42):
    rng = np.random.RandomState(rng_seed)
    bootstrapped_scores = {'auc': [], 'acc': [], 'prec': [], 'rec': [], 'spec': [], 'npv': [], 'f1': [], 'bacc': []}

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue

        y_true_b = y_true[indices]
        y_prob_b = y_prob[indices]
        y_pred_b = y_pred[indices]

        bootstrapped_scores['auc'].append(roc_auc_score(y_true_b, y_prob_b))
        bootstrapped_scores['acc'].append(accuracy_score(y_true_b, y_pred_b))
        bootstrapped_scores['prec'].append(precision_score(y_true_b, y_pred_b, zero_division=0))
        bootstrapped_scores['rec'].append(recall_score(y_true_b, y_pred_b, zero_division=0))
        bootstrapped_scores['f1'].append(f1_score(y_true_b, y_pred_b, zero_division=0))
        bootstrapped_scores['bacc'].append(balanced_accuracy_score(y_true_b, y_pred_b))

        cm = confusion_matrix(y_true_b, y_pred_b, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        bootstrapped_scores['spec'].append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        bootstrapped_scores['npv'].append(tn / (tn + fn) if (tn + fn) > 0 else 0.0)

    cis = []
    for key in ['auc', 'acc', 'prec', 'rec', 'spec', 'npv', 'f1', 'bacc']:
        sorted_scores = np.array(bootstrapped_scores[key])
        sorted_scores.sort()
        cis.append((np.percentile(sorted_scores, 2.5), np.percentile(sorted_scores, 97.5)))

    return cis[0], cis[1], cis[2], cis[3], cis[4], cis[5], cis[6], cis[7]


class DataGenerator_double(Sequence):
    def __init__(self, data_paths, math_paths, batch_size=8, dim=(48, 96, 96, 1), shuffle=True):
        self.data_paths = data_paths
        self.math_paths = math_paths
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        data_paths_temp = [self.data_paths[k] for k in indexes]
        math_paths_temp = [self.math_paths[k] for k in indexes]
        X, M = self.__data_generation(data_paths_temp, math_paths_temp)
        return [X,M], X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, data_paths_temp, math_paths_temp):
        data = read_images_from_single_folders(data_paths_temp)
        math = read_images_from_single_folders(math_paths_temp)

        data = data / 255.0
        math = math / 255.0

        return data, math


def read_images_from_single_folders(folder_paths):
    images = []
    for path_pair in folder_paths:
        folder_images = []
        channel_images = []
        index = 0
        for filename in os.listdir(path_pair):
            if filename.endswith(".bmp"):
                image_path = path_pair + "\\" + str(index) + ".bmp"
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                channel_images.append(image)
            index += 1
        folder_images.append(np.array(channel_images))
        images.append(np.stack(folder_images, axis=-1))
    return np.array(images)
