
import os,sys
from PIL import Image

def batch_rename(folder_name, name):
    """
    :param folder_name (str): the folder's name/path
    :param name(str): what the files are going to renamed as
    :param verbose(boolean): if the function will be verbose or not and give feedback
    :return: none
    """
    files = os.listdir(folder_name)
    os.chdir(folder_name) # go to specified directory

    for i in range(len(files)):    # iterate through each image

        file_path = str(files[i]) # get file's name
        print(file_path)
        extension = (file_path[file_path.index("."):]) # get file's extension
        new_name = str(name) + str("_") + "{:03d}".format(i+1) + str(extension) # create new name,  https://stackoverflow.com/questions/134934/display-number-with-leading-zeros was used for formatting leading 0s
        os.rename(file_path, new_name) # rename the file
    print ("Done!")


def batch_resize(folder_name, height, width,overwrite):
    """
    This function resizes all the images in a given folder to a certain width and height
    :param folder_name (str): the folder's name
    :param height (int): the new height of all the images
    :param width (int): the new width of all the images
    :param overwrite (str): if the resized image will create a duplicate, will be used mostly for debugging
    :return:
    """

    files = os.listdir(folder_name) # get the folder
    os.chdir(folder_name) # go to specified directory

    for i in range(len(files)):    # iterate through each image
        image_name = str(files[i])  # get file's name
        image = Image.open(image_name)

        if overwrite.lower() == 'y': # resized image will save over
            new_image = image.resize((width,height))
            new_image.save(image_name)

        else: # a copy of the resized image will be saved
            new_image = image.resize((width,height))
            new_name = str("resized_") + str(image_name)
            new_image.save(new_name)

    print ("Done!")

    pass

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

        if user_choice == '1':
            folder_path = input("Please enter the folder path")
            new_name = input("Please enter the new name")

            batch_rename(folder_path,new_name)
        if user_choice =='2':
            folder_path = input("Please enter the folder path")
            height = int(input("Please enter the new height"))
            width = int(input("Please enter the new width"))
            batch_resize(folder_path,height,width,'y')
          
        if user_choice =='3':
            return



