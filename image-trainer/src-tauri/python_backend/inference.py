import argparse
import json
import os
import torch
from torchvision import transforms
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='PyTorch Inference Script')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image to test')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to the folder containing best_model.pth and model_metadata.json')
    args = parser.parse_args()

    image_path = args.image_path
    model_dir = args.model_dir

    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    model_path = os.path.join(model_dir, 'best_model.pth')

    if not os.path.exists(image_path):
        print(json.dumps({"status": "error", "message": f"Image not found at {image_path}"}), flush=True)
        return

    if not os.path.exists(metadata_path):
        print(json.dumps({"status": "error", "message": f"Metadata missing at {metadata_path}"}), flush=True)
        return

    if not os.path.exists(model_path):
        print(json.dumps({"status": "error", "message": f"Model weights missing at {model_path}"}), flush=True)
        return

    try:
        # Load Metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        model_name = metadata['model']
        num_classes = metadata['num_classes']
        class_names = metadata['classes']

        # Setup Device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Create Model Architecture
        import model_factory
        model, _ = model_factory.create_model(model_name, num_classes, device)

        # Load Weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Data Transform (equivalent to validation transform)
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Load Image
        image = Image.open(image_path).convert('RGB')
        input_tensor = val_transform(image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Sort by highest confidence
            top_prob, top_class_idx = torch.max(probabilities, 0)
            
            # Create a full dictionary of probabilities for all classes
            class_probs = {class_names[i]: float(probabilities[i]) for i in range(num_classes)}
            
            sorted_probs = dict(sorted(class_probs.items(), key=lambda item: item[1], reverse=True))

        result = {
            "status": "success",
            "prediction": class_names[top_class_idx.item()],
            "confidence": float(top_prob),
            "all_probabilities": sorted_probs
        }

        print(json.dumps(result), flush=True)

    except Exception as e:
        print(json.dumps({"status": "error", "message": f"Inference failed: {str(e)}"}), flush=True)

if __name__ == "__main__":
    main()
