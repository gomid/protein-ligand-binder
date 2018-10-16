from utils import generate
from model import build_model
from keras import optimizers, losses, utils


def training():
    radis = 10
    dimension = radis * 2 + 1
    model = build_model(input_shape=(dimension, dimension, dimension, 3))
    # utils.plot_model(model, to_file='model.png')
    adam = optimizers.Adam()
    model.compile(optimizer=adam, loss=losses.binary_crossentropy, metrics=["accuracy"])

    data = []
    labels = []
    for i in range(1, 3001):
        for j in range(1, 3001):
            grids = generate(i, j, 10)
            data.extend(grids)
            label = 1 if i == j else 0
            labels.extend([label]*(len(grids)))

    model.fit([data], [labels], validation_split=0.2, batch_size=10, epochs=3)


if __name__ == '__main__':
    training()

