from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from tensorflow.keras.losses import BinaryCrossentropy

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load the pre-trained model
model = load_model('models/happysadmodel.h5', compile=False)
model.compile(optimizer='adam', loss=BinaryCrossentropy(reduction='sum_over_batch_size'), metrics=['accuracy'])

def predict_emotion(img_path):
    image = load_img(img_path, target_size=(256, 256))  # Adjust size if needed
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)

    if prediction[0] < 0.5:
        return 'Happy'
    else:
        return 'Sad'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            emotion = predict_emotion(file_path)
            return render_template('index.html', emotion=emotion, image_url=url_for('static', filename=f'uploads/{filename}'))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
