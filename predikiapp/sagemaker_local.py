import json
import sagemaker
from sagemaker.pytorch import PyTorchModel
import boto3
import os
import botocore

from model_sagemaker.sagemaker_env import BUCKET_NAME, ROLE_NAME, ENDPOINT_NAME


def build_upload_model_remote():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build_p = os.path.join(base_dir, 'model_sagemaker')
    os.chdir(build_p)
    cmd_pack = 'tar czf model.tar.gz code sentiment.pth'
    os.system(cmd_pack)
    cmd_upload = 'aws s3 cp model.tar.gz s3://%s' % BUCKET_NAME #overwrites
    os.system(cmd_upload)


def deploy_local():
    #difference to remote: role, instance, model_dir
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='SagemakerLocal')['Role']['Arn']

    print('start deploy')
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(base_dir, 'model_sagemaker/model.tar.gz')
    #source_dir = os.path.join(base_dir, 'model_source')
    pytorch_model = PyTorchModel(model_data=model_dir,
                                 role=role, entry_point='inference.py',
                                 framework_version='1.3.1',
                                 py_version='py3',
                                 #source_dir=source_dir, #'s3://sagemaker-studio-ff9g693z8mk/model.tar.gz'
                                 )
    predictor = pytorch_model.deploy(instance_type='local', initial_instance_count=1)
    print('endpoint_name', predictor.endpoint_name)
    print('done')
    return predictor.endpoint_name


def deploy_remote():
    try:
        role = sagemaker.get_execution_role()
    except ValueError:
        iam = boto3.client('iam')
        role = iam.get_role(RoleName=ROLE_NAME)['Role']['Arn']

    # You can also configure a sagemaker role and reference it by its name.
    # role = "CustomSageMakerRoleName"
    # in model.tar.gz: model.pth and code with inferency.py and requirements.txt
    print('sagemaker vrs should be 2.9.2', sagemaker.__version__)
    model_path = 's3://%s/model.tar.gz' % BUCKET_NAME
    pytorch_model = PyTorchModel(model_data=model_path,
                                 role=role, entry_point='inference.py',
                                 framework_version='1.3.1',
                                 py_version='py3',
                                 )
    predictor = pytorch_model.deploy(instance_type='ml.t2.medium', initial_instance_count=1)
    print('endpoint_name', predictor.endpoint_name)
    print('done')
    return predictor.endpoint_name


def sentiment_predict(text, endpoint_name, local=True):
    #custom_attributes = "c000b4f9-df62-4c85-a0bf-7c525f9104a4"  # An example of a trace ID.
    content_type = 'application/json'  # The MIME type of the input data in the request body.
    accept = 'application/json'  # The desired MIME type of the inference in the response.

    body = {'text': text}
    body_json = json.dumps(body)
    if local:
        client = sagemaker.local.LocalSagemakerRuntimeClient()  #sagemaker.local.LocalSagemakerClient()
    else:
        client = boto3.client('sagemaker-runtime')
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        # CustomAttributes=custom_attributes,
        ContentType=content_type,
        Accept=accept,
        Body=body_json
    )

    resp_body = response['Body']
    if type(resp_body) == botocore.response.StreamingBody:
        resp_body = resp_body.read()
        resp_data = resp_body.decode('utf-8')
    else:
        resp_data = resp_body.data.decode('utf-8')

    print(resp_data)

    return resp_data


if __name__ == '__main__':
    #build_upload_model_remote()
    #res = multi_predict_text('people')
    #endpoint_name = deploy_local()
    #endpoint_name = deploy_remote()
    remote_endpt_name = ENDPOINT_NAME
    #res = sentiment_predict('bad is bad is good. is actually very good. ', endpoint_name)
    res = sentiment_predict('bad is bad is good. is actually very good. ', remote_endpt_name, local=False)
    #print(res)