from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi import HTTPException
import torch
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
import json
import os 
import boto3
import uuid
from utils import helper
import shutil
import base64

app = FastAPI()

# s3 bucket name for storing input images 
BUCKET_NAME = 'food-input-images'

# dynamo db table for storing output
TABLE_NAME = 'food-response-metadata'

# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb_client = boto3.client('dynamodb', region_name="us-east-1")
runtime_client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Uncomment when running locally
# session = boto3.Session(profile_name='hb')
# s3_client = session.client('s3')
# dynamodb_client = session.client('dynamodb')
# runtime_client = session.client("bedrock-runtime", region_name="us-east-1")

# Function to copy model files to /tmp for Lambda - For online deployment
def copy_model_to_tmp():
    if not os.path.exists("/tmp/models"):
        os.makedirs("/tmp/models")
    if not os.path.exists("/tmp/data"):
        os.makedirs("/tmp/data")
    shutil.copy("models/model_v3_kf_20250528_104923.pth", "/tmp/models/")
    shutil.copy("data/class_names_kf.json", "/tmp/data/")

# Copy files to /tmp
copy_model_to_tmp()

# load class names data for model referecne 
with open("/tmp/data/class_names_kf.json", "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)
weights = EfficientNet_B0_Weights.DEFAULT
img_transforms = weights.transforms()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modle_identifier = "model_v3_kf_20250528_104923.pth"

#uncomment when running locally
# model = efficientnet_b0(weights=weights)
# model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=num_classes)
# model.load_state_dict(torch.load("models/model_v3_kf_20250528_104923.pth", map_location=device))
# model.to(device)
# model.eval()

#comment when running locally
model = efficientnet_b0(weights=weights)
model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=num_classes)
model.load_state_dict(torch.load("/tmp/models/model_v3_kf_20250528_104923.pth", map_location=device))
model.to(device)
model.eval()

@app.get("/")
async def root():
    return {"message": "Welcome to the Food-Scanner API!"}

@app.post("/analyze-food")
async def predict_image(image_file: UploadFile = File(...)):
    try:
        if not image_file:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        if not image_file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await image_file.read()

        image_id = str(uuid.uuid4())
    
        # Upload image to S3
        s3_key = f"images/{image_id}_{image_file.filename}"
        s3_client.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=contents)

        temp_path = f"/tmp/temp_{image_file.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)

        result = helper.predict(model=model,
                         image_path=temp_path,
                         class_names=class_names,
                         transform=img_transforms)
        # print(result)
        food_name = result["food_name"]
        confidence_score = result["confidence_score"]

        final_output = helper.query_bedrock(food_name, confidence_score, temp_path)

        json_start = final_output.find('{')
        json_end = final_output.rfind('}') + 1
        json_str = final_output[json_start:json_end]

        food_info = json.loads(json_str)


        dynamodb_client.put_item(
            TableName=TABLE_NAME,
            Item={
                'image_id': {'S': image_id},
                'image_key': {'S': s3_key},
                'nutrition_info': {'S': json.dumps(food_info)}
            }
        )
        os.remove(temp_path)

        return JSONResponse(content={
            "final_output": food_info
            })
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
