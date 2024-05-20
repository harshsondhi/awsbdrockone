import boto3
import json

from similarity import cosineSimilarity
client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")  

facts =[
    'The first computer was invented in the 1940s.',
    'John F. Kennedy was the 35th President of the United States.',
    'The first moon landing was in 1969.',
    'The capital of France is Paris.',
    'Earth is the third planet from the sun.',
]

newFact = 'I like to play computer games'
question = 'Who is the president of USA?'

def getEmbedding(text: str):
    response = client.invoke_model(
        body=json.dumps({
            "inputText": text,
        }),
        modelId='amazon.titan-embed-text-v1',
        contentType='application/json',
        accept='application/json'
    )
    response_body = json.loads(response.get('body').read())
    return response_body.get('embedding')

factsWithEmbedding = []
similarities = []

for fact in facts:
    factsWithEmbedding.append({
        'factText': fact,
        'embedding': getEmbedding(fact)
    })
    
questionEmbedding = getEmbedding(newFact)    


for fact in factsWithEmbedding:
    similarities.append({
        'origText': fact['factText'],
        'similarity': cosineSimilarity(fact['embedding'], questionEmbedding)
    })
    
    
print(f"Similarities for fact: '{newFact}' with:")
similarities.sort(key=lambda x: x['similarity'], reverse=True)
for similarity in similarities:
    print(f"  '{similarity['origText']}': {similarity['similarity']:.2f}")    