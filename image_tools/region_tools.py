from matplotlib import image
import pandas as pd
import cv2
import numpy

REGION_COLOR = (0, 0, 255)  # BGR format
REGION_THICKNESS = 2
REGION_FONT_SIZE = 1.5
REGION_FONT_THICKNESS = 2
REGION_TEXT_MARGIN = 3


# Преобразовывает координаты из 2592x1944 версии в 1000x750
def convert_coordinates(regions: pd.core.frame.DataFrame) -> None:
    for i in range(0, regions.shape[0]):
        regions.loc[i, 'X'] = regions['X'][i] * 1000 // 2592
        regions.loc[i, 'W'] = regions['W'][i] * 1000 // 2592
        regions.loc[i, 'Y'] = regions['Y'][i] * 750 // 1944
        regions.loc[i, 'H'] = regions['H'][i] * 750 // 1944


# Возвращает изображение с отображенными регионами
def display_regions(img: numpy.ndarray, regions: pd.core.frame.DataFrame) -> numpy.ndarray:
    result = img.copy()
    for i in range(0, regions.shape[0]):
        X = regions['X'][i]
        Y = regions['Y'][i]
        SlotID = regions['SlotId'][i]
        width = regions['W'][i]
        height = regions['H'][i]
        cv2.rectangle(result, (X, Y), (X + width, Y + height), REGION_COLOR, REGION_THICKNESS)
        cv2.putText(result, str(SlotID), (X + REGION_TEXT_MARGIN, Y + height - REGION_TEXT_MARGIN),
                    cv2.FONT_HERSHEY_PLAIN, REGION_FONT_SIZE, REGION_COLOR,
                    thickness=REGION_THICKNESS)
    return result


# Пример использования
if __name__ == '__main__':
    img = image.imread('test_data/image.jpg', cv2.IMREAD_COLOR)
    regions = pd.read_csv('test_data/coords.csv', sep=';')
    # convert_coordinates(regions) # раскоментить если координаты не преобразованы
    new_img = display_regions(img, regions)
    cv2.imwrite('output.jpg', new_img)
