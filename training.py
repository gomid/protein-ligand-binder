from utils import generate, read_pdb
from model import build_model, build_classifier
from keras import optimizers, losses, callbacks, models
import os
import time
import psutil
from random import shuffle
import numpy as np


def initializer():
    global proteins
    global ligands
    proteins = [[None, None]]
    ligands = [[None, None]]
    for i in range(1, RANGE):
        p_coordinates, p_atom_types = read_pdb("training_data/{0}_pro_cg.pdb".format('%04d' % i))
        l_coordinates, l_atom_types = read_pdb("training_data/{0}_lig_cg.pdb".format('%04d' % i))
        proteins.append([p_coordinates, p_atom_types])
        ligands.append([l_coordinates, l_atom_types])


def parallel_generate(i):
    # data = []
    # labels = []
    positive = []
    negative = []
    grids = generate(ligands[i][0], ligands[i][1], proteins[i][0], proteins[i][1], RADIUS, DISTANCE_THRESHOLD)
    positive.append((grids, 1))

    # data.extend(grids)
    # label = 1
    # labels.extend([label] * (len(grids)))

    # generate 1 negative example
    randomized_range = list(range(1, RANGE))
    shuffle(randomized_range)
    count = NEGATIVE_EXAMPLE
    for j in randomized_range:
        if i == j:
            continue
        grids = generate(ligands[j][0], ligands[j][1], proteins[i][0], proteins[i][1], RADIUS, DISTANCE_THRESHOLD)
        # data.extend(grids)
        # label = 0
        # labels.extend([label] * (len(grids)))
        if len(grids) > 0:
            negative.append((grids, 0))
            count = count - 1
        if count <= 0:
            # return data, labels
            return positive, negative
    # return data, labels
    return positive, negative


def generate_training_data_parallel():
    import multiprocessing
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cores, initializer)
    print("Generating training examples on {} CPU cores".format(cores))
    # data, labels = zip(*pool.map(parallel_generate, range(1, RANGE)))
    positive, negative = zip(*pool.map(parallel_generate, range(1, RANGE)))
    pool.close()
    pool.join()
    positive = [item for sublist in positive for item in sublist]
    negative = [item for sublist in negative for item in sublist]
    # data = [item for sublist in data for item in sublist]
    # labels = [item for sublist in labels for item in sublist]
    return positive, negative


def prepare_data():
    start_time = time.time()
    # data, labels = generate_training_data_parallel()
    positive, negative = generate_training_data_parallel()

    # data, labels = generate_training_data()
    print("--- %s seconds ---" % (time.time() - start_time))
    # memory usage
    process = psutil.Process(os.getpid())
    print("Used total memory: {}".format(process.memory_info().rss))
    return positive, negative


def flatten(examples):
    data = []
    labels = []
    for ex in examples:
        grids = ex[0]
        data.extend(grids)
        labels.extend([ex[1]] * (len(grids)))
    return data, labels


def train_atom(examples):
    dimension = RADIUS * 2 + 1
    model = build_model(input_shape=(dimension, dimension, dimension, 3))
    # utils.plot_model(model, to_file='model.png')
    # adam = optimizers.Adam(decay=0.01)
    adam = optimizers.Adam()
    model.compile(optimizer=adam, loss=losses.binary_crossentropy, metrics=["accuracy"])
    model.summary()

    data, labels = flatten(examples)
    print("Generated {} examples".format(len(data)))

    print("Starting training")
    model.fit([data], [labels], validation_split=0.2, batch_size=100, epochs=20,
              # class_weight=dict(zip(unique_labels, weights)),
              callbacks=[callbacks.EarlyStopping(patience=3)]
              )

    print("Saving model")
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save("model.h5")

    return model


def train(model_file=None):
    positive, negative = prepare_data()
    examples = positive + negative[:len(positive)]
    shuffle(examples)

    if model_file:
        atom_model = models.load_model(model_file)
    else:
        atom_model = train_atom(examples)


    test_data = positive + negative[len(positive):]
    shuffle(test_data)
    inputs = []
    labels = []
    for ex in test_data:
        probs = atom_model.predict(np.array(ex[0]))
        inputs.append(probs)
        labels.append(ex[1])

    res = [i.mean() for i in inputs]
    print(res)


    # classifier = build_classifier()
    # optimizer = optimizers.Adam()
    # classifier.compile(optimizer=optimizer, loss=losses.binary_crossentropy, metrics=["accuracy"])
    # classifier.summary()
    # classifier.fit([inputs], [labels], validation_split=0.2, batch_size=10, epochs=20,
    #                callbacks=[callbacks.EarlyStopping(patience=3)]
    #                )


if __name__ == '__main__':
    RANGE = 3001
    RADIUS = 10
    DISTANCE_THRESHOLD = 10
    NEGATIVE_EXAMPLE = 2
    # train_atom()
    train()
