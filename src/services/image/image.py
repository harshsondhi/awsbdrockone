import boto3
import json
import base64
from time import time

AWS_REGION_BEDROCK = "us-west-2"
S3_BUCKET = "images-bucket-6165202024"

client = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION_BEDROCK)   
s3_client = boto3.client("s3")


def handler(event, context):
    # print("inside def handler(event, context)---------------------------------- ")
    body = json.loads(event["body"])
    description = body.get("description")
    if description:
        titan_config = get_titan_onfiguration(description)
        response = client.invoke_model(
            body=titan_config, 
            modelId="amazon.titan-image-generator-v1", 
            accept="application/json", 
            contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        base64_image = response_body.get("images")[0]
        signed_url = save_image_to_s3(base64_image)
        return {
            "statusCode": 200,
            "body": json.dumps({"url": signed_url}),        
        }
    return {
        "statusCode": 400,
        "body": json.dumps({"error": "description required!"}),  
    }

def get_titan_onfiguration(description: str):
    return json.dumps(
    {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": description,
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": 512,
            "width": 512,
            "cfgScale": 8.0,
        },
    }
)
    
    
def save_image_to_s3(base84_image: str):
    image_file = base64.b64decode(base84_image)
    image_name = f"image_{int(time())}.png"
    
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=image_name,
        Body=image_file,
        ContentType="image/png"
    ) 
    
    signed_url = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": image_name},
        ExpiresIn=3600
    ) 
    return signed_url
    
    
    