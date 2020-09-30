# filename: inference.py
# def model_fn(model_dir)
#def input_fn(request_body, request_content_type)
#def predict_fn(input_data, model)
#def output_fn(prediction, content_type)
import torch
import os
import json

from simpletransformers.classification import ClassificationModel


def model_fn(model_dir):
    #logger.info('Loading the model.')
    cuda_available = torch.cuda.is_available()
    algorithm = 'xlnet'
    modelFolder = os.path.join(model_dir, 'sentiment.pth')
    args = {
        'output_dir': modelFolder,
        'reprocess_input_data': True,
        'overwrite_output_dir': True,
        'num_train_epochs': 5,
        'silent': True,
        'use_cached_eval_features': False,
    }
    model = ClassificationModel(algorithm, modelFolder, args=args,  # runs torch.load()
                                      use_cuda=cuda_available)

    #logger.info('Done loading model')
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        text_str = input_data['text']
        return text_str
    raise Exception(f'Requested unsupported ContentType in content_type {request_content_type}')


def predict_fn(input_data, model):
    #logger.info('Generating prediction based on input parameters.')
    prediction, raw_outputs = model.predict([input_data])
    print(prediction)
    print(raw_outputs)
    threshold = 0.3
    result = [1 if predictionValue > threshold else 0 for predictionValue in prediction]
    # tend_dict defined in ReplaceSentimentsWithIndexes
    tend_dict = {0: 'negative',  # id = 1
                 1: 'controversial',  # id = 3
                 2: 'positive'}  # id = 2
    pred_int = prediction[0]
    print('prediction is', tend_dict[pred_int])
    pred_d = {'tendency_id': str(pred_int),
              'tendency_name': tend_dict[pred_int]}

    return pred_d


def output_fn(prediction, content_type='application/json'):
    if content_type == 'application/json':
        return json.dumps(prediction)
    raise Exception(f'Requested unsupported ContentType in Accept:{content_type}')



if __name__ == '__main__':
    #res = multi_predict_text('people')
    from RunTrain import BASE_DIR
    _MODEL_SAVE_DEF = os.path.join(BASE_DIR, 'model_data/')
    modeli = model_fn(_MODEL_SAVE_DEF)
    text = 'bad is bad is good. is actually very good. '
    text_json = json.dumps(text)
    content_t = 'application/json'
    inp_data = input_fn(text_json, content_t)
    pred = predict_fn(inp_data, modeli)
    output_d = output_fn(pred, content_t)
    res = json.loads(output_d)

    print(res)
