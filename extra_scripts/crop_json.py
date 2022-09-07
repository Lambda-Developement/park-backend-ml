import json
import sys
from PIL import Image


def crop_json(json_file: str, image_file: str) -> list[Image.Image]:
    # Read json file into dict
    with open(json_file, "r") as f:
        data = json.load(f)
    # Read image file into Image object
    image = Image.open(image_file)
    crops = []

    # Crop image
    for i, annotation in enumerate(data[0]["annotations"]):
        if annotation['label'] == "parking lot":
            x, y, w, h = annotation['coordinates'].values()
            x, y, w, h = int(x), int(y), int(w), int(h)
            crop = image.crop((x, y, x + w, y + h))
            crops.append(crop)

    assert(len(crops) == len(data[0]["annotations"]), "Something went wrong with annotation count...")
    return crops

def crop_json_and_save(json_file: str, image_file: str, output_dir: str) -> None:
    crops = crop_json(json_file, image_file)
    for i, crop in enumerate(crops):
        crop.save(f"{output_dir}/{i}.jpg")

if __name__ == "__main__":
    # Take args from command line
    crop_json_and_save(sys.argv[1], sys.argv[2], sys.argv[3])