import numpy as np
from math import sqrt
from scipy import spatial
import sys


def read_pdb(filename, filter=False):
    # outlier = 0
    with open(filename, 'r') as file:
        strline_L = file.readlines()
        # print(strline_L)
    coordinates = []
    atomtype_list = list()
    for strline in strline_L:
        # removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns
        stripped_line = strline.strip()

        line_length = len(stripped_line)
        # print("Line length:{}".format(line_length))
        if line_length < 78:
            print("ERROR: line length is different. Expected>=78, current={}".format(line_length))

        x = float(stripped_line[30:38].strip())
        y = float(stripped_line[38:46].strip())
        z = float(stripped_line[46:54].strip())
        # if filter and (x > 200 or x < -200 or y > 200 or y < -200 or z > 200 or z < -200):
        #     outlier = outlier+1
        #     continue
        coordinates.append([x, y, z])
        atomtype = stripped_line[76:78].strip()
        if atomtype == 'C':
            atomtype_list.append('h') # 'h' means hydrophobic
        else:
            atomtype_list.append('p') # 'p' means polar

    # if outlier > 0:
    #     print(outlier)
    return np.array(coordinates), atomtype_list


def find_bounding_box(coordinates):
    min_point = coordinates.min(axis=0)
    max_point = coordinates.max(axis=0)
    return min_point, max_point

# Deprecated
def cal_atom_distance(p_atom, l_atom):
    x = p_atom[0] - l_atom[0]
    y = p_atom[1] - l_atom[1]
    z = p_atom[2] - l_atom[2]
    return sqrt(x * x + y * y + z * z)


def max_atom_distance(p_coordinates, l_coordinates):
    dis_matrix = spatial.distance.cdist(p_coordinates, l_coordinates)
    return max(dis_matrix.min(axis=0))


def initializer():
    global data
    data = []
    for i in range(1, 3001):
        p_coordinates, _ = read_pdb("training_data/{0}_pro_cg.pdb".format('%04d' % i))
        l_coordinates, _ = read_pdb("training_data/{0}_lig_cg.pdb".format('%04d' % i))
        data.append([p_coordinates, l_coordinates])

def distance_worker(i):
    distances = []
    for j in range(1, 3001):
        if i == j:
            continue
        dis = max_atom_distance(data[i-1][0], data[j-1][1])
        if dis > 7:
            distances.append(dis)
    return distances

"""
paired : min 2.3, max 7, avg 5
non-paired && distance > 10: max 610, avg 53, 82% out of 9000000 combinations
non-paired && distance > 7: max 610, avg 50.9, 86.8% out of 9000000 combinations
"""
def calculate_atom_distances():
    import multiprocessing
    pool = multiprocessing.Pool(4, initializer)
    distances = pool.map(distance_worker, range(1, 3001))
    pool.close()
    pool.join()
    distances = [item for sublist in distances for item in sublist]
    print(max(distances))
    print(min(distances))
    print(len(distances))
    print(sum(distances))

def preprocess(seq, scale=1):
    grids = []
    for type in ["pro", "lig"]:
        N = 400 * scale
        shift = N / 2 - 1
        coordinates, atom_type_list = read_pdb("training_data/{0}_{1}_cg.pdb".format('%04d' % seq, type))
        grid = np.zeros(shape=(N, N, N, 2), dtype=np.ushort)
        offsets = np.array([shift, shift, shift]).astype(int)
        for i in range(len(coordinates)):
            atom = (scale * coordinates[i]).astype(int) + offsets
            if atom[0] >= N or atom[1] >= N or atom[2] >= N:
                return None, None
            polarity = 0 if atom_type_list[i] == 'C' else 1
            grid[atom[0]][atom[1]][atom[2]][0] = 1
            grid[atom[0]][atom[1]][atom[2]][1] = polarity
        grids.append(grid)
    return grids[0], grids[1]


def generate(lig_id, pro_id, radius, distance_threshold=None):
    """
    Generate grids for protein-lig pair based on each ligand atom
    Round to nearest integer
    """
    def add_to_grid(grid, atom, polarity, target=0):
        grid[atom[0]][atom[1]][atom[2]][0] = 1
        grid[atom[0]][atom[1]][atom[2]][1] = polarity
        grid[atom[0]][atom[1]][atom[2]][2] = target

    grids = []
    lig_atoms, lig_atom_type_list = read_pdb("training_data/{0}_{1}_cg.pdb".format('%04d' % lig_id, "lig"))
    pro_atoms, pro_atom_type_list = read_pdb("training_data/{0}_{1}_cg.pdb".format('%04d' % pro_id, "pro"))

    if distance_threshold and max_atom_distance(pro_atoms, lig_atoms) > distance_threshold:
        return grids

    N = radius * 2 + 1
    offset = np.array([radius, radius, radius])
    for i in range(len(lig_atoms)):
        lig_atom = lig_atoms[i]
        center = (lig_atom + np.array([0.5, 0.5, 0.5])).astype(int)
        grid = np.zeros(shape=(N, N, N, 3))
        lo = center - offset
        hi = center + offset

        add_to_grid(grid, center - lo, 0 if lig_atom_type_list[i] == 'C' else 1, target=1)

        for j in range(len(pro_atoms)):
            atom = (pro_atoms[j] + np.array([0.5, 0.5, 0.5])).astype(int)
            if np.all(atom >= lo) and np.all(atom <= hi):
                add_to_grid(grid, atom - lo, 0 if pro_atom_type_list[j] == 'C' else 1)

        for j in range(len(lig_atoms)):
            atom = (lig_atoms[j] + np.array([0.5, 0.5, 0.5])).astype(int)
            if i != j and np.all(atom >= lo) and np.all(atom <= hi):
                add_to_grid(grid, atom - lo, 0 if lig_atom_type_list[j] == 'C' else 1)

        grids.append(grid)

    return grids


"""
[310.935 432.956 435.107]
[-244.401 -229.648 -177.028]
"""
def coordinate_range(type="pro"):
    maxs = []
    mins = []
    for i in range(1, 3001):
        p_coordinates, _ = read_pdb("training_data/{0}_{1}_cg.pdb".format('%04d' % i, type))
        min_point, max_point = find_bounding_box(p_coordinates)
        maxs.append(max_point)
        mins.append(min_point)
    print(np.array(maxs).max(axis=0))
    print(np.array(mins).min(axis=0))

def test():
    for i in range(1, 2):
        p_coordinates, type = read_pdb("training_data/{0}_pro_cg.pdb".format('%04d' % i), filter=True)
        preprocess(p_coordinates, type)


def main():
    # calculate_atom_distances()
    # np.set_printoptions(threshold=np.nan)
    generate(1, 1, 10)

    # pro, lig = preprocess(1, 1)
    # print(sys.getsizeof(pro))
    # print(sys.getsizeof(lig))

    # x_range_max = 0
    # y_range_max = 0
    # z_range_max = 0
    # a = 0
    # b = 0
    # c = 0
    # for i in range(1, 3001):
    #     p_coordinates, _ = read_pdb("training_data/{0}_pro_cg.pdb".format('%04d' % i))
    #     min_point, max_point = find_bounding_box(p_coordinates)
    #     ranges = max_point - min_point
    #     if ranges[0] > x_range_max:
    #         x_range_max = ranges[0]
    #         a = i
    #     if ranges[1] > y_range_max:
    #         y_range_max = ranges[1]
    #         b = i
    #     if ranges[2] > z_range_max:
    #         z_range_max = ranges[2]
    #         c = i
    #
    # print(x_range_max, y_range_max, z_range_max)
    # print(a, b, c)
    # p_coordinates, _ = read_pdb("training_data/{0}_pro_cg.pdb".format('%04d' % 2797))
    # min_point, max_point = find_bounding_box(p_coordinates)
    # print(min_point, max_point)
    # p_coordinates, _ = read_pdb("training_data/{0}_pro_cg.pdb".format('%04d' % 2719))
    # min_point, max_point = find_bounding_box(p_coordinates)
    # print(min_point, max_point)
    # p_coordinates, _ = read_pdb("training_data/{0}_pro_cg.pdb".format('%04d' % 594))
    # min_point, max_point = find_bounding_box(p_coordinates)
    # print(min_point, max_point)

if __name__ == '__main__':
    main()



