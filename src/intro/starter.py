import boto3
import pprint

bedrock = boto3.client(
    service_name='bedrock',
    region_name='us-west-2',
)

pp = pprint.PrettyPrinter(depth=4)

def list_foundation_models():
    models = bedrock.list_foundation_models()
    for model in models["modelSummaries"]:
        pp.pprint(model)
        pp.pprint("--------------------------")
        
def get_foundation_model(modelIdentifier):
    model = bedrock.get_foundation_model(modelIdentifier=modelIdentifier)
    pp.pprint(model)
    pp.pprint("--------------------------")     
    
list_foundation_models()   
#get_foundation_model('anthropic.claude-3-sonnet-20240229-v1:0')      