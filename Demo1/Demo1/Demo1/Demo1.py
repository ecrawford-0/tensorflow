# Importing all necessary libraries 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense 
from tensorflow.keras import backend as K 



def train_model():
    train_data_dir = 'data/train'
    validation_data_dir = 'data/validate'
    nb_train_samples =400 
    nb_validation_samples = 100
    epochs = 10
    batch_size = 16
    img_width = 200
    img_height = 200


    if K.image_data_format() == 'channels_first': 
        input_shape = (3, img_width, img_height) 
    else: 
        input_shape = (img_width, img_height, 3) 

    model = Sequential() 
    model.add(Conv2D(32, (2, 2), input_shape=input_shape)) 
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2))) 
  
    model.add(Conv2D(32, (2, 2))) 
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2))) 
  
    model.add(Conv2D(64, (2, 2))) 
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2))) 
  
    model.add(Flatten()) 
    model.add(Dense(64)) 
    model.add(Activation('relu')) 
    model.add(Dropout(0.5)) 
    model.add(Dense(1)) 
    model.add(Activation('sigmoid')) 

    model.summary()

    model.compile(loss='binary_crossentropy', 
                  optimizer='rmsprop', 
                  metrics=['accuracy']) 
    train_datagen = ImageDataGenerator( 
        rescale=1. / 255, 
        shear_range=0.2, 
        zoom_range=0.2, 
        horizontal_flip=True) 
  
    test_datagen = ImageDataGenerator(rescale=1. / 255) 
  
    train_generator = train_datagen.flow_from_directory( 
        train_data_dir, 
        target_size=(img_width, img_height), 
        batch_size=batch_size, 
        class_mode='binary') 
  
    validation_generator = test_datagen.flow_from_directory( 
        validation_data_dir, 
        target_size=(img_width, img_height), 
        batch_size=batch_size, 
        class_mode='binary') 
  
    model.fit_generator( 
        train_generator, 
        steps_per_epoch=nb_train_samples // batch_size, 
        epochs=epochs, 
        validation_data=validation_generator, 
        validation_steps=nb_validation_samples // batch_size) 

    model.save('full_model_saved.h5') 


def load_model(old_model):
    
    # Recreate the exact same model, including its weights and the optimizer
    new_model = tf.keras.models.load_model(old_model)

    # Show the model architecture
    new_model.summary()
"""    
    user_choice = input("test with image?(y/n)")
    if user_choice.lower() =='y':
        image = input("Enter image path")
        new_model.predict(image)
"""

   # loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
   # print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

menu = """
1. train model
2. load model
3. exit

"""

while True:
    print(menu)
    user_input = input()

    if user_input == '1':
        train_model()
    if user_input == '2':
        file = ('full_model_saved.h5')
        #file = input("enter the filename")
        load_model(file)
    if user_input =='3':
        break