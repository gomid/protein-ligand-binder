from utils import preprocess
from model import build_model
from keras import optimizers, losses, utils
import numpy as np

def training():
    model = build_model()
    # utils.plot_model(model, to_file='model.png')
    adam = optimizers.Adam()
    model.compile(optimizer=adam, loss=losses.mse, metrics=["accuracy"])
    proteins = []
    ligands = []
    for i in range(1, 100):
        pro, lig = preprocess(i)
        if pro is not None:
            proteins.append(pro)
            ligands.append(lig)

    labels = np.ones(shape=(len(proteins),))

    model.fit([proteins, ligands], [labels], validation_split=0.2, batch_size=10, epochs=3)


if __name__ == '__main__':
    training()

