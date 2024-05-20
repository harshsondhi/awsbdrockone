import boto3
import json
import pprint

#amazon.titan-text-express-v1
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-west-2',
)

#client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")

history =[]

def get_history():
    return "\n".join(history)


titan_model_id = 'amazon.titan-text-express-v1'

def get_configuration(prompt: str):
    titan_config = json.dumps({ 
                  "inputText": get_history(),
                  "textGenerationConfig": {
                            "maxTokenCount": 4096,
                            "stopSequences": [],
                            "temperature": 0,
                            "topP": 1,
                           }
                         
        })
    return titan_config

print(
    "Bot: Hello I am a chatbot. I can help you with anything you want to talk about"
)

while True:
    user_input = input("User: ")
    history.append("User" + user_input)
    if user_input.lower() == "bye":
        print("Bot: Bye! Have a great day!")
        break
    else:
        titan_config = get_configuration(user_input)
        response = bedrock.invoke_model(
            body=titan_config,
            modelId=titan_model_id,
            contentType='application/json',
            accept='application/json'
        )
        response_body = json.loads(response.get('body').read())
        output_text = response_body.get('results')[0].get('outputText').strip()
        print(output_text)
        history.append(output_text)