import sys
import io
from PIL import Image
import cv2
import torch
from flask import Flask, render_template, request, make_response
from werkzeug.exceptions import BadRequest
import os
from werkzeug.datastructures import FileStorage
app = Flask(__name__)
# create a python dictionary for your models d = {<key>: <value>, <key>: <value>, ..., <key>: <value>}
dictOfModels = {}
# create a list of keys to use them in the select part of the html code
listOfKeys = []
def get_prediction(img_bytes,model):
    img = Image.open(io.BytesIO(img_bytes))
    # inference
    results = model(img, size=400)  
    return results
@app.route('/', methods=['GET'])
def get():
  # in the select we will have each key of the list in option
  return render_template("index.html", len = len(listOfKeys), listOfKeys = listOfKeys)
@app.route('/home', methods=['GET'])
def home():
    return render_template("home.html")

@app.route('/home', methods=['POST'])
def predict():
    file = extract_img(request)
    if isinstance(file, str):
        # Handle the error message here
        return file
    img_bytes = file.read()
    # choice of the model
    image_url = '/static/uploaded_image.jpg'
    with open('static/uploaded_image.jpg', 'wb') as f:
        f.write(img_bytes)
    results = get_prediction(img_bytes, selected_model)
    # updates results.imgs with boxes and labels
    results.render()
    
    # Get the detected classes and confidence scores
    class_indices = results.pred[0][:, -1].tolist()
    confidence_scores = results.pred[0][:, 4].tolist()
    
    # Map class indices to class names from results.names
    class_names = [results.names[index] for index in class_indices]
    
    if not confidence_scores:  # Check if confidence_scores is empty
        return render_template('not_found.html')
    
    # Find the class with the highest confidence score
    max_confidence_idx = confidence_scores.index(max(confidence_scores))
    selected_class_name = class_names[max_confidence_idx]
    selected_confidence = confidence_scores[max_confidence_idx]
    
    if selected_class_name == 'altocumulus':
        return render_template('altocumulus.html', image_url=image_url)
    elif selected_class_name == 'altostratus':
        return render_template('altostratus.html', image_url=image_url)
    elif selected_class_name == 'cirrocumulus':
        return render_template('cirrocumulus.html', image_url=image_url)
    elif selected_class_name == 'cirrostratus':
        return render_template('cirrostratus.html', image_url=image_url)
    elif selected_class_name == 'cirrus':
        return render_template('cirrus.html', image_url=image_url)
    elif selected_class_name == 'cumulonimbus':
        return render_template('cumulonimbus.html', image_url=image_url)
    elif selected_class_name == 'cumulus':
        return render_template('cumulus.html', image_url=image_url)
    elif selected_class_name == 'nimbostratus':
        return render_template('nimbostratus.html', image_url=image_url)
    elif selected_class_name == 'stratocumulus':
        return render_template('stratocumulus.html', image_url=image_url)
    else:
        return render_template('not_found.html')

def extract_img(request):
    # checking if image uploaded is valid
    if 'file' not in request.files:
        return render_template("custom_bad_request.html")
    file = request.files['file']
    if not file:
        return render_template("custom_bad_request.html")
    return file

if __name__ == '__main__':
    print('Starting yolov5 webservice...')
    # Getting directory containing models from command args (or default 'models_train')
    models_directory = 'models_train'
    if len(sys.argv) > 1:
        models_directory = sys.argv[1]
    for r, d, f in os.walk(models_directory):
        for file in f:
            if ".pt" in file:
                model_name = os.path.splitext(file)[0]
                model_path = os.path.join(r, file)
                print(f'Loading model {model_path} with path {model_path}...')
                dictOfModels[model_name] = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        for key in dictOfModels:
            listOfKeys.append(key) # put all the keys in the listOfKeys

    # In this case, you can specify the model you want to use directly
    selected_model = dictOfModels.get("best_model")  # Change "model1" to the model you want to use

    # starting app
    app.run(debug=True, host='0.0.0.0')
