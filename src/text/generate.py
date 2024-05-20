import boto3
import json
import pprint

#amazon.titan-text-express-v1
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-west-2',
)

#client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")


titan_model_id = 'amazon.titan-text-express-v1'
llama_model_id = 'meta.llama3-8b-instruct-v1:0'
#meta.llama3-8b-instruct-v1:0

titan_config = json.dumps({ 
                  "inputText": "Tell me a story about a dragon and a knight.",
                  "textGenerationConfig": {
                            "maxTokenCount": 4096,
                            "stopSequences": [],
                            "temperature": 0,
                            "topP": 1,
                           }
                          
        })

llama_config = json.dumps({
    "prompt": "Tell me a story about a dragon",
    "max_gen_len": 512,
    "temperature": 0,
    "top_p": 0.9,
})

response = bedrock.invoke_model(
                                body=titan_config,
                                modelId=titan_model_id,
                                contentType='application/json',
                                accept='application/json'
                                )

llama_response = bedrock.invoke_model(
    body=llama_config,
    modelId=llama_model_id,
    accept="application/json",
    contentType="application/json"
)

response_body = json.loads(response.get('body').read().decode('utf-8'))
llama_response_body = json.loads(llama_response.get('body').read().decode('utf-8'))

pp = pprint.PrettyPrinter(depth=4)
#pp.pprint(response_body.get('results'))

pp.pprint(llama_response_body["generation"]) # llama config