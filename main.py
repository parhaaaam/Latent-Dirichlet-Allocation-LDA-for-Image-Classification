import os

from src.get_images import load_images_from_directory
from src.lda import run_lda
from src.transform_images_into_documents import recognize_elements_in_images

if __name__ == '__main__':

    parent_directory = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of the current script
    image_directory = os.path.join(parent_directory, '', 'images')  # Navigates up to the parent, then to 'Images'

    # Load images
    loaded_images = load_images_from_directory(image_directory)

    # Recognize elements in the loaded images
    recognized_elements = recognize_elements_in_images(loaded_images)

    image_topics, lda_model = run_lda(recognized_elements)

    # Print out the most probable topic for each image
    for img_name, topic in image_topics.items():
        print(f"{img_name}: Topic is {topic}")
