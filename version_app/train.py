import model
import config

from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.models import save_model, load_model
import tensorflow as tf

'''Wipes previous model and retrains on the given dataset'''
def clean_train(test_data, valid_data):
    export_path = "models/model_v" + VERSION
    with open(export_path, 'w') as _:
        pass
    add_train(test_data, valid_data)

'''Trains the model on given dataset'''
def add_train(test_data, valid_data):
    global EPOCH, VERSION, NUM_CLASSES

    # load model
    try:
        tf_model = load_model('models/model_v' + VERSION)
    except:
        tf_model = model.Xception(NUM_CLASSES)

    # FIXME setting learning rate & optimizer
    lr_fn = tf.optimizers.schedules.PolynomialDecay(1e-3, EPOCH, 1e-5, 2)
    opt = tf.optimizers.Adam(lr_fn)

    # train on given test/validation set
    tf_model.compile(optimizer=opt,
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy', 'mse'])

    tf_model.fit(test_data[0], test_data[1], epochs=EPOCH,
                 validation_data=(valid_data[0], valid_data[1]))

    save_tf_model(tf_model)

'''Saves the trained model to a file'''
def save_tf_model(tf_model):
    global VERSION

    export_path = "models/model_v" + VERSION
    save_model(
        tf_model,
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )

'''Classifies data from saved model'''
def predict(data):
    global VERSION
    try:
        tf_model = load_model('models/model_v'+VERSION)
        return tf_model.predict(data)
    except:
        print("Not trained model stored at models/model_v"+VERSION)


if __name__=="__main__":
    config.setup()
    clean_train()