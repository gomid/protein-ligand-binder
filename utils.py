import numpy as np
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


def max_atom_distance(p_coordinates, l_coordinates):
    dis_matrix = spatial.distance.cdist(p_coordinates, l_coordinates)
    return max(dis_matrix.min(axis=0))


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
    if distance_threshold and max_atom_distance(pro_atoms, lig_atoms) > distance_threshold:
        return grids

    N = radius * 2 + 1
    offset = np.array([radius, radius, radius])

    for i in range(len(lig_atoms)):
        grid = np.zeros(shape=(N, N, N, 4))

        center = lig_atoms[i]
        lo = center - offset
        hi = center + offset
        for j in range(len(pro_atoms)):
            atom = pro_atoms[j]
            if np.all(atom >= lo) and np.all(atom <= hi):
                add_to_grid(grid, atom - lo, 0 if pro_atom_type_list[j] == 'h' else 1)

        for j in range(len(lig_atoms)):
            atom = lig_atoms[j]
            if np.all(atom >= lo) and np.all(atom <= hi):
                add_to_grid(grid, atom - lo, 0 if lig_atom_type_list[j] == 'h' else 1, 1)

        grids.append(grid)

    return grids
