
import os,sys
from PIL import Image


def batch_rename(folder_name, name):
    """
    :param folder_name (str): the folder's name/path
    :param name(str): what the files are going to renamed as
    :param verbose(boolean): if the function will be verbose or not and give feedback
    :return: none
    """
    try:
        files = os.listdir(folder_name)
        os.chdir(folder_name) # go to specified directory

        for i in range(len(files)):    # iterate through each image

            file_path = str(files[i]) # get file's name
            print(file_path)
            extension = (file_path[file_path.index("."):]) # get file's extension
            new_name = str(name) + str("_") + "{:03d}".format(i+1) + str(extension) # create new name,  https://stackoverflow.com/questions/134934/display-number-with-leading-zeros was used for formatting leading 0s
            os.rename(file_path, new_name) # rename the file
        print ("Done!")
    except Exception as e:
        print("An error occured", e)

def batch_resize(folder_name, original_dir, height, width,overwrite):
    """
    This function resizes all the images in a given folder to a certain width and height
    :param folder_name (str): the folder's name
    :param original dir (str): the original directory to go back to after resizing everything
    :param height (int): the new height of all the images
    :param width (int): the new width of all the images
    :param overwrite (str): if the resized image will create a duplicate, will be used mostly for debugging
    :return:
    """
    try:
        #print("This is the current directory:", os.getcwd() )
        files = os.listdir(folder_name) # get the folder
        os.chdir(folder_name) # go to specified directory
        
        for i in range(len(files)):    # iterate through each image
            image_name = str(files[i])  # get file's name
            image = Image.open(image_name)

            if overwrite.lower() == 'y': # resized image will save over, will do this by defualt
                new_image = image.resize((width,height))
                new_image.save(image_name)

            else: # a copy of the resized image will be saved
                new_image = image.resize((width,height))
                new_name = str("resized_") + str(image_name)
                new_image.save(new_name)

        os.chdir(original_dir)
        print ("Done!")
        
        
    except Exception as e:
        print("An error occured")

def print_files(path,extension):
    """
    This function prints all the files in a given directory with a certain extension
    :param path(str): the desired file path
    :param extension(str): the desired file extension to look for
    """
    current_dir = os.listdir(path) # gets the directory
    for file in current_dir:
        if ( (file.find(".") != -1) & (file[file.find("."):] == extension)    ): # exclude files and get the desired extension
            print(file) 

menu = """
==================================
1. rename folder's contents
2. resize images 
3. exit
==================================
"""
def show_choices_interface():
    """
    This function shows a menus users can interact with 
    """
    print("welcome, what do you want to do?")
    while True:
        print(menu)
        user_choice = input()
        folder_path = ''
        start_dir = os.getcwd()


        if user_choice == '1':
            
            folder_path = input("\nPlease enter the folder path:\n")          
            if not(os.path.isdir(folder_path)): # if the directory doesn't exist say so 
                print("Directory not found")
                    
            else:   # everything went as excpeted
                 new_name = input("Please enter the new name\n")
                 batch_rename(folder_path,new_name)
            os.chdir(start_dir)

        if user_choice =='2':

           # print(os.listdir('data/train'))
            for folder in os.listdir('data/train'):
                print("data/train/" + str(folder))

            current_dir = os.getcwd()
            folder_path = input("\nPlease enter the folder path:\n")
            height = (input("Please enter the new height:\n"))
            width = (input("Please enter the new width:\n"))
      
            if not (os.path.isdir(folder_path)) or not(height.isnumeric()) or not(width.isnumeric()) : # inputs are not valid
                print("Invalid inputs, make sure diretory exists and height and width are numbers")
           
            else: # inputs are valid
               batch_resize(folder_path,current_dir,int(height),int(width),'y')
            os.chdir(start_dir)
            
        if user_choice =='3':
            return
            


