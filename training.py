from utils import generate, read_pdb
from model import build_model
from keras import optimizers, losses, callbacks
import os
import time
import psutil


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
    for j in range(1, RANGE):
        grids = generate(ligands[j][0], ligands[j][1], proteins[i][0], proteins[i][1], RADIUS, DISTANCE_THRESHOLD, DISTANCE_METRIC)
        data.extend(grids)
        label = 1 if i == j else 0
        labels.extend([label] * (len(grids)))
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


def generate_training_data():
    data = []
    labels = []
    proteins = [[None, None]]
    ligands = [[None, None]]
    for i in range(1, RANGE):
        p_coordinates, p_atom_types = read_pdb("training_data/{0}_pro_cg.pdb".format('%04d' % i))
        l_coordinates, l_atom_types = read_pdb("training_data/{0}_lig_cg.pdb".format('%04d' % i))
        proteins.append([p_coordinates, p_atom_types])
        ligands.append([l_coordinates, l_atom_types])
    for i in range(1, RANGE):
        for j in range(1, RANGE):
            grids = generate(ligands[j][0], ligands[j][1], proteins[i][0], proteins[i][1], RADIUS, DISTANCE_THRESHOLD, DISTANCE_METRIC)
            data.extend(grids)
            label = 1 if i == j else 0
            labels.extend([label]*(len(grids)))
    return data, labels


def training():
    dimension = RADIUS * 2 + 1
    model = build_model(input_shape=(dimension, dimension, dimension, 1))
    # utils.plot_model(model, to_file='model.png')
    adam = optimizers.Adam()
    model.compile(optimizer=adam, loss=losses.binary_crossentropy, metrics=["accuracy"])

    start_time = time.time()
    data, labels = generate_training_data_parallel()
    # data, labels = generate_training_data()
    print("Generated {} examples".format(len(data)))
    print("--- %s seconds ---" % (time.time() - start_time))


    # memory usage
    process = psutil.Process(os.getpid())
    print("Used total memory: {}".format(process.memory_info().rss))

    # print("Starting training")
    # model.fit([data], [labels], validation_split=0.2, batch_size=100, epochs=5, callbacks=[callbacks.EarlyStopping()])
    #
    # print("Saving model")
    # # serialize model to JSON
    # model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # model.save_weights("model.h5")


if __name__ == '__main__':
    RANGE = 20
    RADIUS = 10
    DISTANCE_THRESHOLD = 8
    # DISTANCE_METRIC = 'chebyshev'
    DISTANCE_METRIC = 'euclidean'
    training()

