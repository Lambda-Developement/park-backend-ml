from matplotlib import image
from image_tools.region_tools import convert_coordinates
import pandas as pd
import cv2
import numpy

DESIRED_SIZE = (150, 150)  # Входной размер изображения для нейронки


def crop(img: numpy.ndarray, regions: pd.core.frame.DataFrame, index: int) -> numpy.ndarray:
    '''
    Обрезает изображение, чтобы можно было отправить в нейронку
    :param img: Входное изображение
    :param regions: Массив с координатами парковочных мест
    :param index: Индекс парковочного места в массиве координат
    :return: Обрезанное изображение
    '''
    X = regions['X'][index]
    Y = regions['Y'][index]
    width = regions['W'][index]
    height = regions['H'][index]
    return img[Y:Y + height, X:X + width]


def scale(img: numpy.ndarray) -> numpy.ndarray:
    '''
    Доводит изображения до размера требуемого для нейронки
    :param img: Входное изображение
    :return: Изображение нужного размера
    '''
    return cv2.resize(img, DESIRED_SIZE)


# Пример использования
if __name__ == '__main__':
    img = image.imread('../test_data/image.jpg', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    regions = pd.read_csv('../test_data/coords.csv', sep=';')
    # convert_coordinates(regions) # раскоментить если координаты не преобразованы
    cropped = crop(img, regions, 10)
    new_img = scale(cropped)
    cv2.imwrite('output.jpg', new_img)
