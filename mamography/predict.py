import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os


model = load_model('breast.h5')


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


def predict_image(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    return prediction
    
while True:
    image_dir = input("Enter the directory path containing the images to predict: ").strip()


    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    class_names = ['Benign', 'Malignant']



    for image_file in image_files:
       image_path = os.path.join(image_dir, image_file)
       prediction = predict_image(image_path)
    
   
       class_index = np.argmax(prediction)
       confidence = prediction[0][class_index]
       class_name = class_names[class_index]
    
       print(f"Image: {image_file}")
       print(f"Prediction: {class_name}")
       print(f"Confidence: {confidence:.2f}")
       print()