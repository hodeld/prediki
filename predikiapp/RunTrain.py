#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
from CsvReader import ReadCsv
from HelperFunctions import ShuffleData, ReplaceCategoriesWithIndex, OneHotEncodingForCategories, \
    TransformDataIntoDataframe
from TrainEvalModel import TrainModelForMultiLabel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def trainTheModel():
    """failed on macbook"""
    import numpy as np
    import pandas as pd
    import torch
    import copy
    import sklearn
    import csv

    # CSVReader is optional, we recommend using a database

    # We recommend replacing these two lines with database access and maybe feeding the data as a parameter

    folderPath = os.path.join(BASE_DIR, 'etiki-data')
    (data, categories, tendencies) = ReadCsv(folderPath, 'etikidata.csv', 'companies.csv', 'categories.csv',
                                             'references.csv', 'tendencies.csv', 'topics.csv')

    # IMPORTANT! The input data in this method should be single-labeled, the following functions will convert them.
    rawData = data[:, [13, 4, 13]]

    multiLabelData = OneHotEncodingForCategories(ReplaceCategoriesWithIndex(categories, rawData, True))
    train_df = TransformDataIntoDataframe(multiLabelData)

    amount_of_categories = 5
    algo = 'roberta'
    model_name = 'roberta-base'

    # Change the output directory if you want to save the model else where.
    # These arguments are important for loading the model from disk. Make sure they are in a separate function, which the training and the prediction/loading can access.

    args = {
        'output_dir':  os.path.join(BASE_DIR, 'outputs'),
        'reprocess_input_data': True,
        'overwrite_output_dir': True,
        'num_train_epochs': 6,
        'silent': True,
        'use_cached_eval_features': False,
    }
    model = TrainModelForMultiLabel(algo, model_name, train_df,
                                    amount_of_categories, args, use_cuda=False)  # use_cuda when GPU is available


if __name__ == '__main__':
    trainTheModel()
