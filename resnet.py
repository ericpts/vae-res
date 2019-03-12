import tensorflow.keras as keras


def identitiy(f, transp=False, k=3):

    def fn(input):
        [f1, f2, f3] = f

        if transp:
            conv_layer = keras.layers.Conv2DTranspose
        else:
            conv_layer = keras.layers.Conv2D

        X = conv_layer(f1, kernel_size=1)(input)
        X = keras.layers.BatchNormalization(axis=3)(X)
        X = keras.layers.Activation('relu')(X)

        X = conv_layer(f2, kernel_size=k, padding='same')(X)
        X = keras.layers.BatchNormalization(axis=3)(X)
        X = keras.layers.Activation('relu')(X)

        X = conv_layer(f3, kernel_size=1)(X)
        X = keras.layers.BatchNormalization(axis=3)(X)

        X_input = input

        X = keras.layers.Add()([X, X_input])
        X = keras.layers.Activation('relu')(X)
        return X

    return fn


def convolutional(f, s, transp=False, k=3):

    def fn(input):
        [f1, f2, f3] = f

        if transp:
            conv_layer = keras.layers.Conv2DTranspose
        else:
            conv_layer = keras.layers.Conv2D

        X = conv_layer(f1, kernel_size=1, strides=(s, s))(input)
        X = keras.layers.BatchNormalization(axis=3)(X)
        X = keras.layers.Activation('relu')(X)

        X = conv_layer(f2, kernel_size=k, padding='same')(X)
        X = keras.layers.BatchNormalization(axis=3)(X)
        X = keras.layers.Activation('relu')(X)

        X = conv_layer(f3, kernel_size=1)(X)
        X = keras.layers.BatchNormalization(axis=3)(X)

        X_input = conv_layer(f3, kernel_size=1, strides=(s, s))(input)
        X_input = keras.layers.BatchNormalization(axis=3)(X_input)

        X = keras.layers.Add()([X, X_input])
        X = keras.layers.Activation('relu')(X)
        return X

    return fn
