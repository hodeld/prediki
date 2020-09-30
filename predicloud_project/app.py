# app.py

import json
import os
from simpletransformers.classification import ClassificationModel
import torch
from flask import Flask, request

model = None
app = Flask(__name__)


def load_model():
    global model
    use_cuda = True if torch.cuda.is_available() else False
    print(f"using device: {'cuda' if use_cuda == False else 'cpu'}")
    print(os.getcwd(), os.path.isfile('sentiment.pth/config.json'))
    algorithm = 'xlnet'
    modelFolder = 'sentiment.pth/'
    args = {
        'output_dir': modelFolder,
        'reprocess_input_data': True,
        'overwrite_output_dir': True,
        'num_train_epochs': 5,
        'silent': True,
        'use_cached_eval_features': False,
        'use_multiprocessing': False,  # needed due to bug in simpletransformers in connection with uvicorn
    }
    # global model
    model = ClassificationModel(algorithm, modelFolder, args=args, use_cuda=use_cuda)


@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        #model_name = data.get('model')
        input_data = data['text']
        prediction, raw_outputs = model.predict([input_data])  # runs globally loaded model on the data
        print(prediction)
        print(raw_outputs)
    return json.dumps(raw_outputs.tolist()[0])


if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80)