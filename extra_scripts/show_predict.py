from crop_json import crop_json
from PIL import Image
import sys
import numpy
from image_tools.Classificator import Classificator
from image_tools.Classificator import IMG_SIZE


model = Classificator(model_weights_file = "extra_scripts/image_tools/model_weights.h5")

def predict(crops: list[Image.Image]) -> list[bool]:
    """
        Predicts if a parking lot is occupied or not
    """
    predictions = []
    for crop in crops:
        #  Resize img to IMG_SIZE
        crop = crop.resize(IMG_SIZE)
        # Convert to numpy array
        crop = numpy.array(crop)
        predictions.append(model.make_prediction(crop))

    # Return list of bools (True if occupied, False if not) via map >0.5
    return list(map(lambda x: x.item() > 0.5, predictions))


if __name__ == "__main__":
    # Take args from command line
    json_file, image_file = sys.argv[1], sys.argv[2]
    crops = crop_json(json_file, image_file)
    predictions = predict(crops)
    print(predictions)
    print(f"Occupancy rate of parking lot on {image_file} is {sum(predictions)/len(predictions)} ({sum(predictions)}/{len(predictions)})")