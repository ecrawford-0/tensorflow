import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


def make_prediction():
    
    # dimensions of our images
    img_width, img_height = 200, 200
   
    class_names = ['buff orpington','lavender orpington']
    global test_labels
    test_labels = [0,1]
# load the model we saved
    model = load_model('full_model_saved.h5')
    model.layers[0].input_shape
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    ##use trained model to make a single prediction
    ## Grab an image from the test dataset.
    #img_path = "data/train/Buff Orpington/buff_orpington_013.jpg"

    #img = image.load_img(img_path, target_size=(img_width, img_height))
    #img = image.img_to_array(img)
    #print(img.shape)
    ## Add the image to a batch where it's the only member.
    #img = (np.expand_dims(img,0))

    #print(img.shape)
    #predictions_single = model.predict(img)

    #print(predictions_single)
 
    """
    img_path = "data/train/Buff Orpington/buff_orpington_013.jpg"
       
    # Grab an image from the test dataset.
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img = image.img_to_array(img)
        

    img = (np.expand_dims(img,0))

    print("the shape:",img.shape)
    predictions_single = model.predict(img)

    print("the prediction:",predictions_single)
    plot_value_array(1, predictions_single[0], labels)
    _ = plt.xticks(range(10), class_names, rotation=45)
    np.argmax(predictions_single[0])
    """

    # code from https://gist.github.com/ritiek/5fa903f97eb6487794077cf3a10f4d3e
    # image folder
    folder_path = 'test_data'
    # load all images into a list
    images = []
    for img in os.listdir(folder_path):
        img = os.path.join(folder_path, img)
        img = image.load_img(img, target_size=(img_width, img_height))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        images.append(img)

    # stack up images list to pass for prediction
    images = np.vstack(images)
    classes = model.predict_classes(images, batch_size=10)
    print(classes)
 



def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
