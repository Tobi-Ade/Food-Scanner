"""
Contains functions for training and testing a PyTorch model.
"""
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from tqdm import tqdm
import torch 
import torchvision
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader
from torchinfo import summary 
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import json, os, base64, boto3
import torch
import torchvision
import pickle
from pathlib import Path
from torchvision.models import EfficientNet_B0_Weights


device = "cuda" if torch.cuda.is_available() else "cpu"
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
img_transforms = weights.transforms()


def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform, 
    batch_size: int, 
): 
  train_data = datasets.ImageFolder(train_dir, transform=img_transforms)
  test_data = datasets.ImageFolder(test_dir, transform=img_transforms)

  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
  )

  return train_dataloader, test_dataloader, class_names

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device):

    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device):

    model.eval() 

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:

    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }
    
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results


def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str, 
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device=device):
    
    
    img = Image.open(image_path)

    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


    model.to(device)

    model.eval()
    with torch.inference_mode():
      transformed_image = image_transform(img).unsqueeze(dim=0)
      target_image_pred = model(transformed_image.to(device))

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False);
    plt.show()

def predict(model: torch.nn.Module,
                        image_path: str, 
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device=device):
    
    
    img = Image.open(image_path)

    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


    model.to(device)

    model.eval()
    with torch.inference_mode():
      transformed_image = image_transform(img).unsqueeze(dim=0)
      target_image_pred = model(transformed_image.to(device))

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

 
    return {
        "food_name": class_names[target_image_pred_label],
        "confidence_score": f"{target_image_pred_probs.max():.3f}"
}


# --- SAVING MODEL AND TRANSFORMS ---
def save_model_with_transforms(model, save_dir, model_name):
    """Save model along with the transforms used during training"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = save_dir / model_name
    torch.save(obj=model.state_dict(), f=model_path)
    
    weights = EfficientNet_B0_Weights.DEFAULT
    img_transforms = weights.transforms()
    
    transform_path = save_dir / f"{model_name.split('.')[0]}_transforms.pkl"
    with open(transform_path, 'wb') as f:
        pickle.dump(img_transforms, f)
    
    print(f"Model saved to: {model_path}")
    print(f"Transforms saved to: {transform_path}")
    
    return model_path, transform_path


def query_bedrock(food_name: str, confidence, image_path: str):
    confidence = float(confidence)
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    prompt = (
        f"You are a food analysis and health assessment assistant. "
        f"If the input {food_name} is wrong, analyze internally and output the correct name. NEVER show the user your analysis to get the name of the food"
        f"Analyze the food in this image with complete factual accuracy. "
        f"For all food in the image, guve detailed analysis, even in the name, e.g if there is steak, it could be medium or rare, etc. Such detail should be factored into your analysis, that is how great you are"
        f"Do not hallucinate or include information you're uncertain about. "
        f"For some of the foods passed, especially 'githeri', there may be many variations that do not look exactly like it, internally review when you get input like that anf id it is actually githeri, or whatever food passed, use that for your analysis. Just be aware it may be the same food  just a different variation"
        f"Provide only verifiable nutrition facts.\n\n"
        f"The classified food from the image is: '{food_name}'"
        f"(confidence: {confidence * 100:.1f}%). Use that as context of how confident we are that '{food_name}' is correct.\n\n"
        f"The confidence score and name that you are given is coming from a kenyan food recognition model. The food may not be a kenyan food, if this is the case, Ignore the name and confidence score, and use your abilities as a food expert to generate the name and your own confidence score"
        f"You don't need to inclide information from confidence and food name if yu feel they are not needed"
        f"If the confidence score you are given is below 90%, the food name might still be correct, but you can assume that it is not, in this case, infer the food name and confidence score with your expert analysis capabilities, then use it to generate the additional required information"
        f"infer your own confidence score, however you should alwys be over 90% sure of whatever output you are giving"
        f"Under no circumstance should you tell the user anything about the input {food_name} when it is wrong, do your analysis internally and only give the user the required output. This is super important"
        f"The nutritional analysis should be dependent on the portion of the food in the iage, analye with your expert capabilities and give the right breakdown based on how much the food is"
        
        f"Create a comprehensive nutritional analysis with the following structure:\n\n"

        "1. Food name: Use the passed-in food name directly as the title. There may be other food in the image, if absolutely sure of what they are, include them in the title but never remove the passed in food name, use it as context for other food in the image, if any, else return only the passed in food_name\n"

        "2. Food summary: Short paragraph describing key benefits and health impact. The nutritional breakdown should be based on the size of the portion of food in the image\n"

        "3. Nutritional breakdown:\n"
        "   - Calories: '245kcal'\n"
        "   - Protein: '12g'\n"
        "   - Fat: '8g'\n"
        "   - Carbohydrates: '30g'\n"
        "   - Fiber: '3g'\n"
        "   - Sugar: '5g'\n"
        "   - Sodium: '400mg'\n"

        "4. Ingredients identified: List each ingredient and describe its vitamin/mineral content.\n"

        "5. Health assessment: Is it healthy? Give a factual reason.\n\n"
        "Return result in exactly this JSON format:\n"
        '{'
        '"food_name": "{food_name}", '
        '"confidence_score": "{confidence}", '
        '"food_summary": "<string>", '
        '"nutritional_breakdown": {'
        '  "calories": "<value>kcal", '
        '  "protein": "<value>g", '
        '  "fat": "<value>g", '
        '  "carbohydrates": "<value>g", '
        '  "fiber": "<value>g", '
        '  "sugar": "<value>g", '
        '  "sodium": "<value>mg"'
        '}, '
        '"ingredients_identified": ['
        '  {'
        '    "name": "<ingredient_name>", '
        '    "nutrition_description": "<vitamin_and_mineral_content_description>"'
        '  }'
        '], '
        '"Health_assessment": {"is_healthy?": <true/false>, "Reason": "<string>"}'
        '}'

        "Ensure all numerical values are realistic and factual. If you cannot determine a precise value for any nutritional element, provide a realistic estimate based on similar foods but indicate this in your response."
        "Ensure nice formatting of the output"
    )

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",  
                        "data": encoded_image
                    }}
                ]
            }
        ],
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    client = boto3.client("bedrock-runtime")
    response = client.invoke_model(
        modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )
    response_body = json.loads(response["body"].read())
    formatted_response = response_body.get('content')[0].get('text')
    return formatted_response
