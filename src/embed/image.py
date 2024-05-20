
import boto3
import json
import base64


from similarity import cosineSimilarity
client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")

images = [
    '1.png',
    '2.png',
    '3.png', 
]

def get_image_embedding(image_path: str):
    with open(image_path, "rb") as f:
        inputImage = base64.b64encode(f.read()).decode('utf-8')

    response = client.invoke_model(
        body=json.dumps({
            "inputImage": inputImage,
        }),
        modelId='amazon.titan-embed-image-v1',
        contentType='application/json',
        accept='application/json'
    )

    response_body = json.loads(response.get('body').read())
    return response_body.get('embedding')


imagesWithEmbeddings = []
similarities = []

for image in images:
    imagesWithEmbeddings.append({
        "imageNmae": image,
        "embedding": get_image_embedding(image)
    })
    
testImage = 'cat.png'    

testImageEmbedding = get_image_embedding(testImage)

for image in imagesWithEmbeddings:
    similarities.append({
        "imageName": image.get('imageNmae'),
        "similarity": cosineSimilarity(image.get('embedding'), testImageEmbedding)
    })
    
    
similarities.sort(key=lambda x: x['similarity'], reverse=True)

print(f"Similarities of '{testImage}' with:")
for similarity in similarities:
    print(f"  '{similarity['imageName']}': {similarity['similarity']:.2f}")    