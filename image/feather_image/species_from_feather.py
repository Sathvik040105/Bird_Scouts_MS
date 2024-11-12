from image.feather_image.inference_feather import predict_image_class

def get_species_from_feather(image):
    """
    Get the species of the bird from the feather image.
    """
    return predict_image_class(image)