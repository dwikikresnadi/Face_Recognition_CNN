import keras
# from tensorflow.keras.preprocessing import image
# import pathlib
import cv2
from PIL import Image, ImageOps
import numpy as np

def teachable_machine_classification(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 48, 48, 1), dtype=np.float32)
    image = img
    #image sizing
    size = (48, 48)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #grayscaling image
    grayscale = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2GRAY)
    #turn the image into a numpy array
    image_array = np.asarray(grayscale)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    img_fix = np.expand_dims(normalized_image_array, axis=0)

    # run the inference
    prediction = model.predict(img_fix)
    return np.argmax(prediction) # return position of the highest probability