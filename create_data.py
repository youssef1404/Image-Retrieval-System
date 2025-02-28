import os
from os import listdir

from os.path import isfile, join
import shutil

# Path of the folders containing images
source_path = os.path.join(os.getcwd(), 'data')

# Destination path where to copy images
dest_path = os.path.join(source_path, 'images')

for folder in os.listdir(source_path):

    # Join the source path and folder to get full folder path
    folder_path = os.path.join(source_path, folder)

    # Reset count for each folder
    image_count = 0

    # Get list of files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for i in range(50):
        # Copy file
        file_path = os.path.join(folder_path, files[i])
        dest_file = os.path.join(dest_path, files[i])
        shutil.copy(file_path, dest_file)

        # Increment count
        image_count += 1