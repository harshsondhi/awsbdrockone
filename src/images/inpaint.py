import boto3
import json
import base64

client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")

def get_configuration(inputImage: str):
    return json.dumps({
    "taskType": "INPAINTING",
    "inPaintingParams": {
        "text": "Make the cat disappear from the image.", 
        "negativeText": "bad quality, low res",
        "image": inputImage,
        "maskPrompt": "cat"
    },
    "imageGenerationConfig": {
        "numberOfImages": 1,
        "height": 512,
        "width": 512,
        "cfgScale": 8.0,
    }
})




# stability_image_config = json.dumps({
#     "taskType": "INPAINTING",
#     "inpaintingParams": {
#         "text": "Make the cat disappear from the image.", 
#         "negativeText": "cat",
#         "image": inputImage    
#     },
#     "imageGenerationConfig": {
#         "numberOfImages": 1,
#         "height": 512,
#         "width": 512,
#         "cfgScale": 8.0,
#     }
# })

with open("mycat.png", "rb") as f:
    inputImage = base64.b64encode(f.read()).decode('utf-8')

response = client.invoke_model(
    body=get_configuration(inputImage),
    modelId="amazon.titan-image-generator-v1",
    contentType='application/json',
    accept='application/json'
)

response_body = json.loads(response.get('body').read())
base4_image = response_body.get('images')[0]

base_64_image = base64.b64decode(base4_image)
file_path="mycat.png"
with open(file_path, "wb") as f:
    f.write(base_64_image)

