# -*- coding: utf-8 -*-

import os
import pdb
import pickle
import numpy as np
import itertools
import datetime

from glob import glob


def get_time():
    return datetime.datetime.now()


def total_time_elapsed(start, finish):
    elapsed = finish - start

    total_seconds = int(elapsed.total_seconds())
    total_minutes = int(total_seconds // 60)
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    seconds = int(round(total_seconds % 60))

    return "{0:02d}+{1:02d}:{2:02d}:{3:02d} [Seconds elapsed: {4}]".format(elapsed.days, hours, minutes, seconds, elapsed)


def safe_create_dir(dirname):
    try:
        os.makedirs(dirname)
    except OSError:
        pass


def retrieve_fnames(input_path, file_types):

    folders = []
    for root, dirs, files in os.walk(input_path, followlinks=True):
        for f in files:
            if os.path.splitext(f)[1].lower() in file_types:
                folders += [root]
                break

    folders = sorted(folders)

    all_fnames = []
    for folder in folders:
        fnames = [glob(os.path.join(folder, '*' + file_type)) for file_type in file_types]
        fnames = sorted(list(itertools.chain.from_iterable(fnames)))
        all_fnames += fnames

    return all_fnames

def resize_img_keeping_aspect_ratio(img, min_axis):
    ratio = min_axis / np.min(img.shape)
    n_rows, n_cols = img.shape[:2]
    new_n_rows = int(n_rows * ratio)
    new_n_cols = int(n_cols * ratio)

    if len(img.shape) == 3:
        new_shape = (new_n_rows, new_n_cols, img.shape[2])
    else:
        new_shape = (new_n_rows, new_n_cols)

    return np.resize(img, new_shape)


def save_object(obj, fname):

    try:
        os.makedirs(os.path.dirname(fname))
    except OSError:
        pass

    fo = open(fname, 'wb')
    pickle.dump(obj, fo)
    fo.close()


def load_object(fname):
    fo = open(fname, 'rb')
    obj = pickle.load(fo)
    fo.close()

    return obj


def balanced_accuracy(ground_truth, predicted_labels):
    """ Compute the Balanced accuracy

    Args:
        ground_truth (numpy.ndarray): The ground-truth.
        predicted_labels (numpy.ndarray): The predicted labels.

    Returns:
        float: Return the balanced accuracy, whose value can varies between 0.0 and 1.0, which means the worst and the
        perfect classification results, respectively.

    """
    categories = np.unique(ground_truth)

    bal_acc = []
    for cat in categories:
        idx = np.where(ground_truth == cat)[0]
        y_true_cat, y_pred_cat = ground_truth[idx], predicted_labels[idx]
        tp = [1 for k in range(len(y_pred_cat)) if y_true_cat[k] == y_pred_cat[k]]
        tp = np.sum(tp)
        bal_acc += [tp / float(len(y_pred_cat))]

    bal_acc = np.array(bal_acc)

    return np.mean(bal_acc)
