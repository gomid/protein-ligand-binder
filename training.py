from utils import generate, read_pdb
from model import build_model
from keras import optimizers, losses, callbacks
import os
import time
import psutil


def generate_training_data():
    X = []
    Y = []
    for i in range(1, RANGE):
        for j in range(1, RANGE):
            grids, labels = generate(i, j, RADIUS, DISTANCE_THRESHOLD)
            X.extend(grids)
            Y.extend(labels)
    return X, Y


def training():
    dimension = RADIUS * 2 + 1
    model = build_model(input_shape=(dimension, dimension, dimension, 2))
    # utils.plot_model(model, to_file='model.png')
    adam = optimizers.Adam(decay=0.01)
    model.compile(optimizer=adam, loss=losses.mse, metrics=["accuracy"])

    start_time = time.time()
    data, labels = generate_training_data()
    # data, labels = generate_training_data()
    print("Generated {} examples".format(len(data)))
    print("--- %s seconds ---" % (time.time() - start_time))
    # memory usage
    process = psutil.Process(os.getpid())
    print("Used total memory: {}".format(process.memory_info().rss))

    print("Starting training")
    model.fit([data], [labels], validation_split=0.2, batch_size=100, epochs=50,
              callbacks=[callbacks.EarlyStopping(patience=2)]
              )

    print("Saving model")
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")


if __name__ == '__main__':
    RANGE = 3001
    RADIUS = 7
    DISTANCE_THRESHOLD = 7
    training()

