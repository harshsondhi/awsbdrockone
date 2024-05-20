import boto3
import json
clent = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")

fact = "Fist moon landing was in 1969"
animal = "cat"

response = clent.invoke_model(
    body=json.dumps({
        "inputText": animal,
    }),
    modelId='amazon.titan-embed-text-v1',
    contentType='application/json',
    accept='application/json'
)

response_body = json.loads(response.get('body').read())
print(response_body.get('embedding'))