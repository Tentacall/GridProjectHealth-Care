from flask import Flask, render_template, request, jsonify
import tempfile
import os
from utils.preprocessing import prepare_image
from models.Q_model_simple import SimpleQClassifier
import torch.multiprocessing as mp
import torch

app = Flask(__name__)

device = torch.device("cpu")
mp.set_start_method('spawn')
model_name = "model_quantam_v3.0.19"
model = SimpleQClassifier(device)
loadpath = f"models/checkpoint/{model_name}.pth"
model.load_state_dict(torch.load(loadpath))
model.eval()

@app.route('/')
def main():
    return render_template("index.html")

@app.route('/result', methods=['POST'])
def result():
    try:
        # Check if the 'image' file is in the request
        if 'image' in request.files:
            image_file = request.files['image']
            temp_dir = tempfile.gettempdir()
            temp_image_path = os.path.join(temp_dir, image_file.filename)
            image_file.save(temp_image_path)
            
            print(temp_image_path)
            image = prepare_image(temp_image_path, transform_image = True)
            image = image.unsqueeze(0)
            image = image.to(device)
            with torch.no_grad():
                outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            # Get the size of the uploaded image
            
            return jsonify({'size': predicted[0].item()})
        else:
            return jsonify({'error': 'No image file in the request'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
