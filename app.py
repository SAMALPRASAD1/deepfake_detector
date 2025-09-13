from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the model
model = tf.keras.models.load_model('model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    filename = ""
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load and preprocess the image
            img = cv2.imread(filepath)
            img = cv2.resize(img, (256, 256))  # Resize to match model input
            img = img / 255.0  # Normalize pixel values if required
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Predict
            prediction = model.predict(img)
            print(prediction)  # For debugging

            if prediction[0][0] > 0.5:
                result = "Fake Image"
            else:
                result = "Real Image"

    return render_template('index.html', result=result, filename=filename)

if __name__ == "__main__":
    app.run(debug=True)
