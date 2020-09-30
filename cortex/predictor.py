# predictor.py
import json

from simpletransformers.classification import ClassificationModel
import torch
from transformers import pipeline
from starlette.responses import JSONResponse  # on cortexlabs included


class PythonPredictor:
    def __init__(self, config):
        device = 0 if torch.cuda.is_available() else -1
        print(f"using device: {'cuda' if device == 0 else 'cpu'}")
        algorithm = 'xlnet'
        modelFolder = 'sentiment.pth'
        args = {
            'output_dir': modelFolder,
            'reprocess_input_data': True,
            'overwrite_output_dir': True,
            'num_train_epochs': 5,
            'silent': True,
            'use_cached_eval_features': False,
            'use_multiprocessing': False,  # needed due to bug in simpletransformers in connection with uvicorn
        }
        model_pre = ClassificationModel(algorithm, modelFolder, args=args, use_cuda=False)

        self.sentiment = model_pre
        # from example
        #self.analyzer = pipeline(task='sentiment-analysis', device=device)
        #self.summarizer = pipeline(task='summarization', device=device)

    def predict(self, query_params, payload):
        model_name = query_params.get('model')
        input_data = payload['text']
        if model_name == 'sentiment':
            print('start sentiment prediction', input_data)
            prediction, raw_outputs = self.sentiment.predict([input_data])
            print(prediction)
            print(raw_outputs)
            return json.dumps(raw_outputs.tolist()[0])

        if model_name == "sentiment-analysis":
            return self.analyzer(payload["text"])[0]
        elif model_name == "summarizer":
            summary = self.summarizer(payload["text"])
            return summary[0]["summary_text"]
        else:
            return JSONResponse({"error": f"unknown model: {model_name}"}, status_code=400)


if __name__ == '__main__':
    predictor = PythonPredictor(config=None)
    q_params = {'model': 'sentiment'}
    pload = {'text': 'hallo'}

    predictor.predict(q_params, pload)
