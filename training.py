from utils import generate, read_pdb
from model import build_model
from keras import optimizers, losses, callbacks
import os
import time
import psutil
from random import shuffle


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
    print("Loaded training dataset")


def parallel_generate(i):
    data = []
    labels = []
    grids = generate(ligands[i][0], ligands[i][1], proteins[i][0], proteins[i][1], RADIUS, DISTANCE_THRESHOLD)
    data.extend(grids)
    label = 1
    labels.extend([label] * (len(grids)))

    # generate 1 negative example
    randomized_range = list(range(1, RANGE))
    shuffle(randomized_range)
    for j in randomized_range:
        if i == j:
            continue
        grids = generate(ligands[j][0], ligands[j][1], proteins[i][0], proteins[i][1], RADIUS, DISTANCE_THRESHOLD)
        data.extend(grids)
        label = 0
        labels.extend([label] * (len(grids)))
        if len(grids) > 0:
            return data, labels
    return data, labels


def generate_training_data_parallel():
    import multiprocessing
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cores, initializer)
    print("Generating training examples on {} CPU cores".format(cores))
    data, labels = zip(*pool.map(parallel_generate, range(1, RANGE)))
    pool.close()
    pool.join()
    data = [item for sublist in data for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    return data, labels


def training():
    dimension = RADIUS * 2 + 1
    model = build_model(input_shape=(dimension, dimension, dimension, 3))
    # utils.plot_model(model, to_file='model.png')
    adam = optimizers.Adam(decay=0.01)
    model.compile(optimizer=adam, loss=losses.binary_crossentropy, metrics=["accuracy"])

    start_time = time.time()
    data, labels = generate_training_data_parallel()
    # data, labels = generate_training_data()
    print("Generated {} examples".format(len(data)))
    print("--- %s seconds ---" % (time.time() - start_time))
    # memory usage
    process = psutil.Process(os.getpid())
    print("Used total memory: {}".format(process.memory_info().rss))

    print("Starting training")
    model.fit([data], [labels], validation_split=0.2, batch_size=100, epochs=50,
              # class_weight=dict(zip(unique_labels, weights)),
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

