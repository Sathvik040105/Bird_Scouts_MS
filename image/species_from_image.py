import random
from image.inference import predict_image_class
# This is just for trial.
# This function needs to be replaced!
def get_species_from_image(image):
    """
    Get the species of the bird from the image.
    """
    # return random.choice(["American Robin", "Bald Eagle", "Blue Jay", "Northern Cardinal", "Red-tailed Hawk", "Ruby-throated Hummingbird", "Tufted Titmouse"])
    return predict_image_class(image)
