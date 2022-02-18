import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMAGE_SIZE = (150, 150)


def make_model(input_shape, num_classes):
    """
        Определение модели для классификации
    """
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = inputs

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


def make_prediction(image: str) -> float:
    """
    Определяет, насколько занята парковка по картинке размера IMAGE_SIZE
    Использовать только для тестов!
    :param image: путь до картинки размером IMAGE_SIZE(например: "test_img/test_busy_2.jpg")
    :return: float - чем ближе к 0, тем вероятнее, что парковка занята.
    """
    model = make_model(input_shape=IMAGE_SIZE + (3,), num_classes=2)
    model.load_weights("model_weights.h5")
    img = keras.preprocessing.image.load_img(
        image, target_size=IMAGE_SIZE
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = predictions[0]
    return score


print(make_prediction("test_img/test_free_2.jpg"))
