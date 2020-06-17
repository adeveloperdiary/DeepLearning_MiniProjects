import tensorflow as tf


def AlexNetModel(num_classes):
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(input_shape=(227, 227, 3), filters=96, kernel_size=(11, 11), strides=(4, 4), padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.0002),
                               kernel_initializer=tf.initializers.he_normal()),
        tf.keras.layers.ReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Dropout(rate=0.25),

        tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0002),
                               kernel_initializer=tf.initializers.he_normal()),
        tf.keras.layers.ReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Dropout(rate=0.25),

        tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0002),
                               kernel_initializer=tf.initializers.he_normal()),
        tf.keras.layers.ReLU(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0002),
                               kernel_initializer=tf.initializers.he_normal()),
        tf.keras.layers.ReLU(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0002),
                               kernel_initializer=tf.initializers.he_normal()),
        tf.keras.layers.ReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Dropout(rate=0.25),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(units=4096, kernel_regularizer=tf.keras.regularizers.l2(0.0002), kernel_initializer=tf.initializers.he_normal()),
        tf.keras.layers.ReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.5),

        tf.keras.layers.Dense(units=4096, kernel_regularizer=tf.keras.regularizers.l2(0.0002), kernel_initializer=tf.initializers.he_normal()),
        tf.keras.layers.ReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.5),

        tf.keras.layers.Dense(units=num_classes, kernel_regularizer=tf.keras.regularizers.l2(0.0002), kernel_initializer=tf.initializers.he_normal()),
        tf.keras.layers.Softmax(dtype='float32')
    ])

    return model
