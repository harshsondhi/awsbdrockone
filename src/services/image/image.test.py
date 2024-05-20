from image import handler
import json

event = {
    "body": json.dumps({"description": "A beautiful sunset over the mountains."})
}

print("response = handler(event, {})----------")
response = handler(event, {})

print(response)