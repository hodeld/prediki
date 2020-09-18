import os

import torch

from RunTrain import BASE_DIR
from simpletransformers.classification import MultiLabelClassificationModel, ClassificationModel

_MODEL_DEF = os.path.join(BASE_DIR, 'model_data/')


def get_category(prediction):
    pred_dict_m = {2: 'Animals',
                   3: 'Environment',
                   1: 'People',
                   4: 'Politics',
                   5: 'Products & Services'}
    for k, p in enumerate(prediction[0], 1):
        if p == 1:
            break
    return pred_dict_m[k]


def multi_predict_text(text_m):
    #Ideally the model wouldn't be loaded everytime someone predicts something, but kept in the GPU memory.
    modelFolder = os.path.join(_MODEL_DEF, 'multi')

    algorithm = 'roberta'
    args = {
        'output_dir': modelFolder,
        'reprocess_input_data': True,
        'overwrite_output_dir': True,
        'num_train_epochs': 6,
        'silent': True,
        'use_cached_eval_features': False,
        'threshold': 0.5
    }
    model_saved = MultiLabelClassificationModel(algorithm, modelFolder, args=args, use_cuda=False)

    prediction, raw_outputs = model_saved.predict([text_m, ])
    print(text_m[:10])
    print(prediction)

    cat = get_category(prediction)
    print(cat)

    threshold = 0.3
    result = [1 if predictionValue > threshold else 0 for predictionValue in prediction[0]]
    return result


def sentiment_predict(text):
    algorithm = 'xlnet'
    modelFolder = os.path.join(_MODEL_DEF, 'sentiment')
    args = {
        'output_dir': modelFolder,
        'reprocess_input_data': True,
        'overwrite_output_dir': True,
        'num_train_epochs': 5,
        'silent': True,
        'use_cached_eval_features': False,
    }
    model_saved = ClassificationModel(algorithm, modelFolder, args=args, use_cuda=False)

    prediction, raw_outputs = model_saved.predict([text])
    print(prediction)
    print(raw_outputs)
    threshold = 0.3
    result = [1 if predictionValue > threshold else 0 for predictionValue in prediction]
    # tend_dict defined in ReplaceSentimentsWithIndexes
    tend_dict = {0: 'negative', #id = 1
                 1: 'controversial', #id = 3
                 2: 'positive'} # id = 2
    pred_int = prediction[0]
    print('prediction is', tend_dict[pred_int])

    return result


if __name__ == '__main__':
    #res = multi_predict_text('people')
    res = sentiment_predict('bad is bad is good. is actually very good. but they also killed people.')
    print(res)