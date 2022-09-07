from matplotlib import image
from image_tools.utility_region import convert_coordinates
import pandas as pd
import cv2
import numpy

DESIRED_SIZE = (150, 150)  # Входной размер изображения для нейронки


# def crop(img: numpy.ndarray, regions: pd.core.frame.DataFrame, index: int) -> numpy.ndarray:
#     '''
#     Обрезает изображение, чтобы можно было отправить в нейронку
#     :param img: Входное изображение
#     :param regions: Массив с координатами парковочных мест
#     :param index: Индекс парковочного места в массиве координат
#     :return: Обрезанное изображение
#     '''
#     X = regions['X'][index]
#     Y = regions['Y'][index]
#     width = regions['W'][index]
#     height = regions['H'][index]
#     return img[Y:Y + height, X:X + width]

def crop(img: numpy.ndarray, X:int, Y:int, W:int, H:int) -> numpy.ndarray:
    '''
    Обрезает изображение, чтобы можно было отправить в нейронку
    :param img: Входное изображение
    :param regions: Массив с координатами парковочных мест
    :param index: Индекс парковочного места в массиве координат
    :return: Обрезанное изображение
    '''
    X = X
    Y = Y
    width = W
    height = H
    return img[Y:Y + height, X:X + width]


def crop(img: numpy.ndarray, regions: pd.core.frame.DataFrame, slot_id: int) -> numpy.ndarray:
    '''
    Измененный метод crop, заточенный под обрезку с определенного слота
    :param img: Входное изображение
    :param regions: Массив с координатами парковочных мест
    :param slot_id: Slot_id парковочного места в массиве координат
    :return: Обрезанное изображение
    '''
    index = 0
    for region in regions.iterrows():
        if region[1]["SlotId"]==slot_id:
            break
        index+=1
    if index==regions.shape[0]:
        type(region)
        raise Exception(f"[crop] : SlotId is not on regions list. ({index})")
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
    cropped = crop(img, regions, 144)
    new_img = scale(cropped)
    cv2.imwrite('output.jpg', new_img)
