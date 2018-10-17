from utils import generate
from model import build_model
from keras import optimizers, losses, utils


def training():
    radius = 10
    dimension = radius * 2 + 1
    model = build_model(input_shape=(dimension, dimension, dimension, 3))
    # utils.plot_model(model, to_file='model.png')
    adam = optimizers.Adam()
    model.compile(optimizer=adam, loss=losses.binary_crossentropy, metrics=["accuracy"])

    data = []
    labels = []
    for i in range(1, 3001):
        for j in range(1, 3001):
            grids = generate(i, j, radius, distance_threshold=10)
            data.extend(grids)
            label = 1 if i == j else 0
            labels.extend([label]*(len(grids)))

    model.fit([data], [labels], validation_split=0.2, batch_size=10, epochs=1)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")


if __name__ == '__main__':
    training()

