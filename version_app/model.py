import tensorflow as tf


class Xception(tf.python.keras.Model):

    def __init__(self, classes=2):
        super.__init__()
        self.classes = classes

    def entry_flow(self, x):
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        # SeparableConv2D Residual Block Chain
        for i in [128, 256, 728]:
            # Residual Block
            residual = x
            residual = tf.keras.layers.Conv2D(filters=i, kernel_size=(1, 1), stride=2, padding='same')(residual)
            residual = tf.keras.layers.BatchNormalization()(residual)

            # SeparableConv2D x2
            for j in range(2):
                if i != 128 or j != 0:
                    x = tf.keras.layers.ReLU()(x)
                x = tf.keras.layers.SeparableConv2D(filters=i, kernel_size=(3, 3), padding='same')(x)
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')

            # Add residual block
            x = tf.keras.layers.add()([x, residual])

        return x

    def middle_flow(self, x):
        for _ in range(8):
            # Residual Block
            residual = x

            # SeparableConv2D x3
            for _ in range(3):
                x = tf.keras.layers.ReLU()(x)
                x = tf.keras.layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same')(x)
                x = tf.keras.layers.BatchNormalization()(x)

            # Add residual block
            x = tf.keras.layers.add()([x, residual])


    def exit_flow(self, x):
        # Residual Block
        residual = x
        residual = tf.keras.layers.Conv2D(filters = 1024, kernel_size=(1,1), strides=2, padding='same')(residual)
        residual = tf.keras.layers.BatchNormalization()(residual)

        # SeparableConv2D x2
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.SeparableConv2D(filters=1024, kernel_size=(3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        # Add residual
        x = tf.kears.layers.add()([x, residual])

        # SeparableConv2D x2
        x = tf.keras.layers.SeparableConv2D(filters=1536, kernel_size=(3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.SeparableConv2D(filters=2048, kernel_size=(3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        # Final pooling
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # Classify
        x = tf.keras.layers.Dense(self.classes, activation=tf.nn.softmax)

    def call(self, x):
        return self.exit_flow(self.middle_flow(self.entry_flow(x)))