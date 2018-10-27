import numpy as np
from math import sqrt
from scipy import spatial


def read_test_pdb(filename):
    with open(filename, 'r') as file:
        strline_L = file.readlines()
    # print(strline_L)

    coordinates = []
    atomtype_list = list()
    for strline in strline_L:
        # removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns
        stripped_line = strline.strip()
        # print(stripped_line)

        splitted_line = stripped_line.split('\t')

        x = float(splitted_line[0])
        y = float(splitted_line[1])
        z = float(splitted_line[2])
        coordinates.append([x, y, z])
        atomtype_list.append(str(splitted_line[3]))

    return np.array(coordinates), atomtype_list


def read_pdb(filename):
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
        coordinates.append([x, y, z])
        atomtype = stripped_line[76:78].strip()
        if atomtype == 'C':
            atomtype_list.append('h') # 'h' means hydrophobic
        else:
            atomtype_list.append('p') # 'p' means polar

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
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cores, initializer)
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
        grid = np.zeros(shape=(N, N, N, 3), dtype=np.ushort)
        offsets = np.array([shift, shift, shift]).astype(int)
        for i in range(len(coordinates)):
            atom = (scale * coordinates[i]).astype(int) + offsets
            if atom[0] >= N or atom[1] >= N or atom[2] >= N:
                return None, None
            polarity = 0 if atom_type_list[i] == 'h' else 1
            grid[atom[0]][atom[1]][atom[2]][0] = 1
            grid[atom[0]][atom[1]][atom[2]][1] = polarity
        grids.append(grid)
    return grids[0], grids[1]


def generate(lig_atoms, lig_atom_type_list, pro_atoms, pro_atom_type_list, radius, distance_threshold=None):
    """
    Generate grids for protein-lig pair based on each ligand atom
    Round to nearest integer
    """
    def add_to_grid(grid, atom, polarity, ligand=0):
        x_y_z = np.rint(atom).astype(int)
        x = x_y_z[0]
        y = x_y_z[1]
        z = x_y_z[2]
        if ligand == 1:
            grid[x][y][z][0] = 1
            grid[x][y][z][1] = polarity
        else:
            grid[x][y][z][2] = 1
            grid[x][y][z][3] = polarity

    grids = []
    # lig_atoms, lig_atom_type_list = read_pdb("training_data/{0}_{1}_cg.pdb".format('%04d' % lig_id, "lig"))
    # pro_atoms, pro_atom_type_list = read_pdb("training_data/{0}_{1}_cg.pdb".format('%04d' % pro_id, "pro"))

    if distance_threshold and max_atom_distance(pro_atoms, lig_atoms) > distance_threshold:
        return grids

    N = radius * 2 + 1
    offset = np.array([radius, radius, radius])

    for i in range(len(lig_atoms)):
        center = lig_atoms[i]

        # center = (lig_atom + np.array([0.5, 0.5, 0.5])).astype(int)
        # center = np.rint(lig_atom).astype(int)

        grid = np.zeros(shape=(N, N, N, 4))

        lo = center - offset
        hi = center + offset
        for j in range(len(pro_atoms)):
            atom = pro_atoms[j]
            # atom = (pro_atoms[j] + np.array([0.5, 0.5, 0.5])).astype(int)
            if np.all(atom >= lo) and np.all(atom <= hi):
                add_to_grid(grid, atom - lo, 0 if pro_atom_type_list[j] == 'h' else 1)

        for j in range(len(lig_atoms)):
            atom = lig_atoms[j]
            # atom = (lig_atoms[j] + np.array([0.5, 0.5, 0.5])).astype(int)
            if np.all(atom >= lo) and np.all(atom <= hi):
                add_to_grid(grid, atom - lo, 0 if lig_atom_type_list[j] == 'h' else 1, 1)

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
    list = []
    for i in range(1, 3001):
        l_coordinates, type = read_pdb("training_data/{0}_lig_cg.pdb".format('%04d' % i))
        list.append(len(l_coordinates))
    list = np.array(list)
    from scipy import stats
    print(stats.describe(list))

def main():
    # calculate_atom_distances()
    test()

if __name__ == '__main__':
    # with open('test_ground_truth.txt', 'w') as f:
    #     f.write('pro_id\tlig_id\n')
    #     for i in range(1, 3001):
    #         f.write('{0}\t{1}\n'.format(i, i))
    a = np.array([[3, 2 ,1], [3, 4, 5]])
    b = np.argsort(a, axis=1)[:, -1:]
    print(b)
    for idx, val in enumerate(b):
        print(a[idx][val])
