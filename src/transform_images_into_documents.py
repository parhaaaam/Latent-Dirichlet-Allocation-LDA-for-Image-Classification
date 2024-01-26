
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np

# Load the pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet')


def recognize_elements_in_images(images):
    elements = {}
    for filename, img in images.items():
        # Convert the image to a numpy array and add the batch dimension
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess the image for the InceptionV3 model
        img_array = preprocess_input(img_array)

        # Predict the top 5 elements
        predictions = model.predict(img_array)
        top_predictions = decode_predictions(predictions, top=10)[0]

        # Store the top predictions
        elements[filename] = [(pred[1], pred[2]) for pred in top_predictions]

    return elements


