import tensorflow as tf

class CNN_Network(tf.python.keras.Model):

    def __init__(self):
        super.__init__()

    def forward(self, x):
        x = tf.keras.layers.Conv2D(filters = 32, kernel_size=(2,2), strides=2, padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters = 64, kernel_size=(2,2), strides=2, padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding='valid')(x)


    def call(self, x):
        return self.forward(x)