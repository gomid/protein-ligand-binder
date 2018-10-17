from utils import generate, read_pdb
from model import build_model
from keras import optimizers, losses, utils


def initializer():
    global proteins
    global ligands
    proteins = [[None, None]]
    ligands = [[None, None]]
    for i in range(1, 3001):
        p_coordinates, p_atom_types = read_pdb("training_data/{0}_pro_cg.pdb".format('%04d' % i))
        l_coordinates, l_atom_types = read_pdb("training_data/{0}_lig_cg.pdb".format('%04d' % i))
        proteins.append([p_coordinates, p_atom_types])
        ligands.append([l_coordinates, l_atom_types])


def parallel_generate(i):
    data = []
    labels = []
    for j in range(1, 3001):
        grids = generate(ligands[j][0], ligands[j][1], proteins[i][0], proteins[i][1], 10, 10)
        data.extend(grids)
        label = 1 if i == j else 0
        labels.extend([label] * (len(grids)))
    return [data, labels]


def generate_training_data():
    import multiprocessing
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cores, initializer)
    print("Generating data on {} CPU cores".format(cores))
    result = pool.map(parallel_generate, range(1, 3001))
    pool.close()
    pool.join()
    result = [item for sublist in result for item in sublist]
    return result[0], result[1]


def training():
    radius = 10
    dimension = radius * 2 + 1
    model = build_model(input_shape=(dimension, dimension, dimension, 3))
    # utils.plot_model(model, to_file='model.png')
    adam = optimizers.Adam()
    model.compile(optimizer=adam, loss=losses.binary_crossentropy, metrics=["accuracy"])

    # data = []
    # labels = []
    # for i in range(1, 3001):
    #     for j in range(1, 3001):
    #         grids = generate(i, j, radius, distance_threshold=10)
    #         data.extend(grids)
    #         label = 1 if i == j else 0
    #         labels.extend([label]*(len(grids)))
    data, labels = generate_training_data()

    model.fit([data], [labels], validation_split=0.2, batch_size=1, epochs=1)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")


if __name__ == '__main__':
    training()

