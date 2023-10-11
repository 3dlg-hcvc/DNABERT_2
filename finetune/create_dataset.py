import argparse
import os

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import StratifiedShuffleSplit


def extract_clean_barcode_list(barcodes):
    barcode_list = []

    for i in barcodes:
        barcode_list.append(str(i[0][0]))

    return barcode_list


def extract_clean_barcode_list_for_aligned(barcodes):
    barcodes = barcodes.squeeze().T
    barcode_list = []
    for i in barcodes:
        barcode_list.append(str(i[0]))

    return barcode_list


def load_data(args):
    x = sio.loadmat(args.input_path)

    if args.using_aligned_barcode:
        barcodes = extract_clean_barcode_list_for_aligned(x["nucleotides_aligned"])
    else:
        barcodes = extract_clean_barcode_list(x["nucleotides"])
    labels = x["labels"].squeeze() - 1

    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_index = None
    val_index = None
    for train_split, val_split in stratified_split.split(barcodes, labels):
        train_index = train_split
        val_index = val_split

    x_train = np.array([barcodes[i] for i in train_index])
    x_val = np.array([barcodes[i] for i in val_index])
    y_train = np.array([labels[i] for i in train_index])
    y_val = np.array([labels[i] for i in val_index])

    number_of_classes = np.unique(labels).shape[0]

    return x_train, y_train, x_val, y_val, barcodes, labels, number_of_classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", dest="input_path", help="path to input data (res101.mat)")
    parser.add_argument("--using_aligned_barcode", default=False, action="store_true")
    parser.add_argument("--output", dest="output", help="path to output folder")
    args = parser.parse_args()

    x_train, y_train, x_val, y_val, *_ = load_data(args)

    # train
    train_df = pd.DataFrame({"sequence": x_train, "label": y_train})
    train_df.to_csv(os.path.join(args.output, "train.csv"), index=False)
    
    # val
    val_df = pd.DataFrame({"sequence": x_val, "label": y_val})
    val_df.to_csv(os.path.join(args.output, "dev.csv"), index=False)
