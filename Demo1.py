# Importing all necessary libraries 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping

import random
import os 
import image_prep 

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array 
from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense 
from tensorflow.keras import backend as K 


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# global variables for training the model
global train_dir, validation_dir, batch_size, total_epochs, verbosity,class_len,class_list
    
train_dir = 'data/train'
validation_dir = 'data/validate'
batch_size = 32
total_epochs = 15
verbosity = 1 

class_len = (len(os.listdir(train_dir))) # how many classes are being catagorized,

class_list = []    # create a blank list for the classes to be catagorized
dir = os.listdir(train_dir) # go into the directory 
for file in dir:# loop through each folder in the directory
    if file.find(".") == -1: # only take the folders
       class_list.append(file)  #add the folder to the list
       
# the different text menus
menu = """
MAIN MENU
===============================
1. prepare images
2. train model
3. make prediction
4. exit
===============================
"""

training_menu = """
TRAINING
===============================
1. show options
2. change options
3. train model
4. exit
===============================
"""

"""
def train_model():
    
    This trains a model and saves it 
    
    global train_dir, validate_dir,batch_size, total_epochs, verbosity,class_len,class_list
    
    train_data_dir = train_dir
    validation_data_dir = validate_dir

    nb_train_samples =400 # how many samples are in the training directories
    nb_validation_samples = 100 # how many samples are in the validation directories
    epochs = total_epochs # number of 'training sessions' the more the higher the accuracy will be
    batch_size = batch_size
    img_width = 200  
    img_height = 200
    num_classes = class_len # how many classes are being catagorized,

    if K.image_data_format() == 'channels_first': 
        input_shape = (3, img_width, img_height) 
    else: 
        input_shape = (img_width, img_height, 3) 

    # building the model
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
    model.add(Dense(num_classes)) 
    model.add(Activation('softmax')) 

    model.summary() # get its summary

    model.compile(loss='categorical_crossentropy', 
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
        classes = class_list,
        class_mode='categorical') 
  
    validation_generator = test_datagen.flow_from_directory( 
        validation_data_dir, 
        target_size=(img_width, img_height), 
        batch_size=batch_size, 
        classes = class_list,
        class_mode='categorical') 
  
    model.fit_generator( 
        train_generator, 
        steps_per_epoch=nb_train_samples // batch_size, 
        epochs=epochs, 
        validation_data=validation_generator, 
        validation_steps=nb_validation_samples // batch_size) 
    
    print("done!")
    file = input("What do you want to call the model?")

    model.save(str(file) + str(".h5"))

def Load_model(old_model):
    
    # Recreate the exact same model, including its weights and the optimizer
    new_model = tf.keras.models.load_model(old_model)
    new_model.layers[0].input_shape
    new_model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    

    # Show the model architecture
    new_model.summary()

    user_choice = input("test with image?(y/n)")

   
    if user_choice.lower() =='y':
        #use trained model to make a single prediction
        # Grab an image from the test dataset.
        img_path = "data/train/Buff Orpington/buff_orpington_013.jpg"
        img_width, img_height = 200, 200


        img = image.load_img(img_path, target_size=(img_width, img_height))

        img = image.img_to_array(img)
        print(img.shape)
        # Add the image to a batch where it's the only member.
        img = (np.expand_dims(img,0))

        print(img.shape)
        predictions_single = new_model.predict(img)

        print("The Prediction is:", predictions_single)
"""
def make_prediction():
    
    # dimensions of our images
    img_width, img_height = 200, 200
   
    # load the model we saved
    print_files(os.getcwd(),".h5")
    try:
        model_path = input("\nWhat model do you want to use?") + str(".h5")
        model = load_model(model_path)
        model.layers[0].input_shape
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        
    
        # code from https://gist.github.com/ritiek/5fa903f97eb6487794077cf3a10f4d3e
        # image folder
        folder_path = 'test_data'
        # load all images into a list
        images = []
        image_names = []
        for img in os.listdir(folder_path):
           
            image_names.append(img[:img.find("_")])
            img = os.path.join(folder_path, img)
            img = image.load_img(img, target_size=(img_width, img_height))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            images.append(img)

        # stack up images list to pass for prediction
        images = np.vstack(images)

        prediction = model.predict_classes(images, batch_size=10)
       
        #model.evaluate()
        classes = {} # create dictionary to store class number with name
        path = os.listdir('data/train')
        for i in range(len(path)):
            classes.update({i:path[i]})    
        print("The predictions:")
        count = 0
        correct_guesses = 0
        for i in prediction:

          print('class:',image_names[count], '\t-\t guess:',classes.get(i))
          if image_names[count] == classes.get(i):
                correct_guesses +=1
          count +=1
            
        print("total correct",(correct_guesses/len(prediction))*100,"%")
    except Exception as e:
        print("There was an error",e )
   
def print_files(path,extension):
    """
    This function prints all the files in a given directory with a certain extension
    :param path(str): the desired file path
    :param extension(str): the desired file extension to look for
    """
    current_dir = os.listdir(path) # gets the directory
    for file in current_dir:
        if ( (file.find(".") != -1) & (file[file.find("."):] == extension)    ): # exclude files and get the desired extension
            print(file) # print the file

def train_CNN(train_directory,validation_directory,target_size=(200,200), classes=None, batch_size=128,num_epochs=20,num_classes=3,verbose=2,model_name='model.h5'):
    """
    Trains a conv net for the flowers dataset with a 5-class classifiction output
    Also provides suitable arguments for extending it to other similar apps
    
    Arguments:
            train_directory: The directory where the training images are stored in separate folders.
                            These folders should be named as per the classes.
            target_size: Target size for the training images. A tuple e.g. (200,200)
            classes: A Python list with the classes 
            batch_size: Batch size for training
            num_epochs: Number of epochs for training
            num_classes: Number of output classes to consider
            verbose: Verbosity level of the training, passed on to the `fit_generator` method
            model_name
    Returns:
            A trained conv net model
    
    """
    from tensorflow.keras.optimizers import RMSprop
    
    # ImageDataGenerator object instance with scaling
    train_datagen = ImageDataGenerator(rescale=1/255)
    validate_datagen = ImageDataGenerator(rescale=1/255)
   
    # Flow training images in batches using the generator
    train_generator = train_datagen.flow_from_directory(
            train_directory,  # This is the source directory for training images
            target_size=target_size,  # All images will be resized to 200 x 200
            batch_size=batch_size,
            # Specify the classes explicitly
            classes = classes,
            # Since we use categorical_crossentropy loss, we need categorical labels
            class_mode='categorical',
            shuffle=True)

    # get images for the validation part
    validation_generator = validate_datagen.flow_from_directory( 
        validation_directory, 
        target_size=target_size, 
        batch_size=batch_size, 
        classes = classes,
        class_mode='categorical',
        shuffle=True) 

    
    input_shape = tuple(list(target_size)+[3])
    
    # Model architecture
    model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 200x 200 with 3 bytes color
   
   # The first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(2, 2),
    

    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),

    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    


    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),

    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
 

    # Flatten the results to feed into a dense layer
    tf.keras.layers.Flatten(),
    # 512 neuron in the fully-connected layer
    tf.keras.layers.Dense(512, activation='relu'),
    # 5output neurons for 5 classes with the softmax activation
    tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Optimizer and compilation
    model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])
    
    # Total sample count
    total_sample=train_generator.n
    

   # Training
    model.fit_generator(
        train_generator, 
        steps_per_epoch=int(total_sample/batch_size),  
        epochs=num_epochs,
        verbose=verbose,
        validation_data =validation_generator,
        validation_steps=int(total_sample/batch_size))

    print("done!")
    model.save(model_name)
    
def train_menu():
    global train_dir, validation_dir, batch_size, total_epochs, verbosity,class_len,class_list
    while True:
        print(training_menu)
        user_input = input()

        if user_input == '1': # show the options
            
            print("training image directory:", train_dir)
            print("validate image directory:", validation_dir)
            print("batch size:",batch_size)
            print("epochs:",total_epochs)
            print("verbosity:",verbosity)
            print("number of classes:", class_len)
            print("classes:",class_list)
         
                   
        if user_input == '2': # change something
          
            print("""
CHANGE OPTIONS
===============================
1. train directory
2. validate directory
3. batch size
4. epochs
5. verbosity
===============================
""")
            print("What option do you want to change?")
            user_choice =input()

            if user_choice =='1': # image directory

                new_val = input("What do you want the new value to be?")
                train_dir = new_val
            if user_choice =='2': # image directory
                new_val = input("What do you want the new value to be?")
                validation_dir = new_val
            if user_choice =='3': # batch size
                
                new_val = int(input("What do you want the new value to be?"))
                batch_size = new_val
            if user_choice =='4': # epochs

                new_val = int(input("What do you want the new value to be?"))
                total_epochs = new_val
            if user_choice =='5': # verbosity

               new_val = int(input("What do you want the new value to be(0, 1, or 2)?"))
               if 0 <= new_val <= 3 :
                    verbosity = new_val 
                             
           
        if user_input == '3': # run the training function
            name=input("What do you want to call the model?") + str(".h5")
            train_CNN(train_dir,validation_dir,target_size=(200,200), classes=class_list, batch_size=batch_size,num_epochs=total_epochs,num_classes=class_len,verbose=verbosity,model_name=name)
            
        if user_input =='4': # go back to the main menu
            break

while True:
    print(menu)
    user_input = input()
    if user_input == '1': # bring image prep menu
       image_prep.show_choices_interface()
    if user_input == '2': # bring training menu up 
        train_menu()
         
    if user_input == '3': # make a prediction
       make_prediction()

    if user_input =='4': # end program
        break

"""
    if user_input =='4':
        
        train_directory = train_dir
        class_len = (len(os.listdir('data/train'))) # how many classes are being catagorized,

        class_list = []    # create a blank list for the classes to be catagorized
        dir = os.listdir(train_directory) # go into the directory 
        for file in dir:# loop through each folder in the directory
            if file.find(".") == -1: # only take the folders
                class_list.append(file)  #add the folder to the list
  

        train_CNN(train_directory=train_directory,classes=class_list,batch_size=32,num_epochs=15,num_classes=class_len,verbose=1)
"""
  
    