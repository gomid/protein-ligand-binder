from utils import generate, read_pdb, read_test_pdb
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
    positive = []
    negative = []
    grids = generate(ligands[i][0], ligands[i][1], proteins[i][0], proteins[i][1], RADIUS, DISTANCE_THRESHOLD)
    positive.append((grids, 1))

    # generate negative example
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
    return positive, negative


def prepare_data():
    start_time = time.time()
    # data, labels = generate_training_data_parallel()
    positive, negative = generate_training_data_parallel()
    print("Generated {0} positive and {1} negative examples".format(len(positive), len(negative)))

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


def train(examples):
    dimension = RADIUS * 2 + 1
    model = build_model(input_shape=(dimension, dimension, dimension, 3))
    # utils.plot_model(model, to_file='model.png')
    # adam = optimizers.Adam(decay=0.01)
    adam = optimizers.Adam()
    model.compile(optimizer=adam, loss=losses.binary_crossentropy, metrics=["accuracy"])
    model.summary()

    data, labels = flatten(examples)
    print("Training on {} examples".format(len(data)))

    print("Starting training")
    model.fit([data], [labels], validation_split=VALIDATION_SPLIT, batch_size=100, epochs=20,
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


def evaluate(model_file=None):
    if model_file:
        atom_model = models.load_model(model_file)
    else:
        positive, negative = prepare_data()
        examples = positive + negative[:len(positive)]
        shuffle(examples)
        atom_model = train(examples)

    # test_data = positive + negative[len(positive):]
    # print("Validating overall result on {} examples".format(len(test_data)))
    # inputs = []
    # labels = []
    # for ex in test_data:
    #     probs = atom_model.predict(np.array(ex[0]))
    #     inputs.append(probs)
    #     labels.append(ex[1])
    #
    # res = [int(round(i.mean())) for i in inputs]
    # result = np.array(res) - np.array(labels)
    # print(np.count_nonzero(result))
    predict(atom_model)


def predict(model):
    test_pro = [[None, None]]
    test_lig = [[None, None]]
    scores = np.zeros(shape=(TEST_RANGE, TEST_RANGE))

    # predict top 10 matching ligands for each protein
    print("Generating test data")
    for i in range(101, 300):
        p_coordinates, p_atom_types = read_test_pdb("training_data/{0}_pro_cg.pdb".format('%04d' % i))
        l_coordinates, l_atom_types = read_test_pdb("training_data/{0}_lig_cg.pdb".format('%04d' % i))
        test_pro.append([p_coordinates, p_atom_types])
        test_lig.append([l_coordinates, l_atom_types])

    print("Testing")
    # candidate_count = 0
    for i in range(1, TEST_RANGE):
        for j in range(1, TEST_RANGE):
            grids = generate(test_lig[j][0], test_lig[j][1], test_pro[i][0], test_pro[i][1], RADIUS, DISTANCE_THRESHOLD)
            if len(grids) > 0:
                # candidate_count = candidate_count + 1
                # FIXME use average atom score as overall score?
                scores[i][j] = model.predict(np.array(grids)).mean()
                print("Evaluated Protein {0} with Ligand {1}, score: {2}".format(i, j, scores[i][j]))

    print("#################################################################################################")
    # print("Evaluated {} candidates".format(candidate_count))

    np.savetxt('test_scores_max.txt', scores.max(axis=1), fmt='%.2f')
    np.savetxt('test_scores_mean.txt', scores.mean(axis=1), fmt='%.2f')

    header_column = np.arange(1, TEST_RANGE).reshape(TEST_RANGE - 1, 1)
    result = np.array([np.argpartition(arr, -10)[-10:] for arr in scores[1:]]).astype(int)
    result = np.append(header_column, result, axis=1)
    print("Saving to text_predictions.txt")
    with open('test_predictions.txt', 'w') as f:
        f.write('pro_id\tlig1_id\tlig2_id\tlig3_id\tlig4_id\tlig5_id\tlig6_id\tlig7_id\tlig8_id\tlig9_id\tlig10_id\n')
        np.savetxt(f, result, fmt='%i', delimiter='\t')


if __name__ == '__main__':
    RANGE = 3001
    RADIUS = 10
    DISTANCE_THRESHOLD = 10
    NEGATIVE_EXAMPLE = 2
    TEST_RANGE = 200
    VALIDATION_SPLIT = 0.1
    # train_atom()
    evaluate("v1.h5")
