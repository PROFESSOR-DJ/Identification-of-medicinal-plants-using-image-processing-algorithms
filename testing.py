import os
from flask import Flask, render_template, request
from flask import jsonify
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model = load_model('crop.h5')
model.load_weights('crop_weights.h5')




def process_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  
    return img




def predict_crop_species(img):
    class_labels = ['Alpinia Galanga (Rasna)',
                    'Amaranthus Viridis (Arive-Dantu)',
                    'Artocarpus Heterophyllus (Jackfruit)',
                    'Azadirachta Indica (Neem)',
                    'Basella Alba (Basale)',
                    'Brassica Juncea (Indian Mustard)',
                    'Carissa Carandas (Karanda)',
                    'Citrus Limon (Lemon)',
                    'Ficus Auriculata (Roxburgh fig)',
                    'Ficus Religiosa (Peepal Tree)',
                    'Hibiscus Rosa-sinensis',
                    'Jasminum (Jasmine)',
                    'Mangifera Indica (Mango)',
                    'Mentha (Mint)',
                    'Moringa Oleifera (Drumstick)',
                    'Muntingia Calabura (Jamaica Cherry-Gasagase)',
                    'Murraya Koenigii (Curry)',
                    'Nerium Oleander (Oleander)',
                    'Nyctanthes Arbor-tristis (Parijata)',
                    'Ocimum Tenuiflorum (Tulsi)',
                    'Piper Betle (Betel)',
                    'Plectranthus Amboinicus (Mexican Mint)',
                    'Pongamia Pinnata (Indian Beech)',
                    'Psidium Guajava (Guava)',
                    'Punica Granatum (Pomegranate)',
                    'Santalum Album (Sandalwood)',
                    'Syzygium Cumini (Jamun)',
                    'Syzygium Jambos (Rose Apple)',
                    'Tabernaemontana Divaricata (Crape Jasmine)',
                    'Trigonella Foenum-graecum (Fenugreek)']  
    predictions = model.predict(img)
    predicted_class = class_labels[np.argmax(predictions)]
    return predicted_class


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        img = process_image(file_path)
        prediction = predict_crop_species(img)
        return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)
