import random
from image.bird_image.inference import predict_image_class

# This is just for trial.
# This function needs to be replaced!
def get_species_from_image(image):
    """
    Get the species of the bird from the image.
    """
    return predict_image_class(image)
