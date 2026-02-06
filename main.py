from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os



# create app
app = Flask(__name__)



# load trained model
model = load_model('models/model.h5')

# Class Labels
class_labels = ['glioma','meningioma','notumor','pituitary']

# define the uploads folder
UPLOAD_FOLDER = "./uploads/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)



# predictor function
def predict_tumor(image_path, model):
  try:
    # load image
    image = load_img(image_path, target_size = (128, 128))
    image_array = img_to_array(image) / 255.0                # normalize pixel values
    image_array = np.expand_dims(image_array, axis = 0)      # adding batch dimensions

    # prediction
    prediction = model.predict(image_array)
    predicted_class_index = np.argmax(prediction)
    confidence_score = np.max(prediction, axis = 1)[0]

    # determine the class
    if class_labels[predicted_class_index] == "notumor":
      return "No Tumor", confidence_score

    else:
      return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

  except Exception as e:
    print("Error in processing the image: ", str(e))



# Route for the main page (index.html)
@app.route("/", methods=['GET','POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']

        if file:
            # save the file
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            # Predict the result
            result, confidence = predict_tumor(file_location, model)

            # return result with image path for display
            return render_template('index.html', result = result, confidence = f"{confidence*100:.2f}", file_path = f"/uploads/{file.filename}")
    
    return render_template('index.html', result = None)



# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# python main
if __name__ == '__main__':
   app.run(debug=True)