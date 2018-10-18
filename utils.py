import numpy as np
from scipy import spatial


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


def max_atom_distance(p_coordinates, l_coordinates):
    """
    paired : min 2.3, max 7, avg 5
    non-paired && distance > 10: max 610, avg 53, 82% out of 9000000 combinations
    non-paired && distance > 7: max 610, avg 50.9, 86.8% out of 9000000 combinations
    """
    dis_matrix = spatial.distance.cdist(p_coordinates, l_coordinates)
    return max(dis_matrix.min(axis=0))


def generate(lig_id, pro_id, radius, distance_threshold=None):
    """
    Generate grids for protein-lig pair based on each ligand atom
    Round to nearest integer
    """
    def add_to_grid(grid, atom, polarity):
        grid[atom[0]][atom[1]][atom[2]][0] = 1
        grid[atom[0]][atom[1]][atom[2]][1] = polarity

    grids = []
    labels = []
    lig_atoms, lig_atom_type_list = read_pdb("training_data/{0}_{1}_cg.pdb".format('%04d' % lig_id, "lig"))
    pro_atoms, pro_atom_type_list = read_pdb("training_data/{0}_{1}_cg.pdb".format('%04d' % pro_id, "pro"))

    if distance_threshold and max_atom_distance(pro_atoms, lig_atoms) > distance_threshold:
        return grids

    N = radius * 2 + 1
    offset = np.array([radius, radius, radius])
    for i in range(len(lig_atoms)):
        lig_atom = lig_atoms[i]
        center = (lig_atom + np.array([0.5, 0.5, 0.5])).astype(int)
        grid = np.zeros(shape=(N, N, N, 2))
        lo = center - offset
        hi = center + offset
        for j in range(len(pro_atoms)):
            atom = (pro_atoms[j] + np.array([0.5, 0.5, 0.5])).astype(int)
            if np.all(atom >= lo) and np.all(atom <= hi):
                add_to_grid(grid, atom - lo, 0 if pro_atom_type_list[j] == 'h' else 1)

        for j in range(len(lig_atoms)):
            atom = (lig_atoms[j] + np.array([0.5, 0.5, 0.5])).astype(int)
            if i != j and np.all(atom >= lo) and np.all(atom <= hi):
                add_to_grid(grid, atom - lo, 0 if lig_atom_type_list[j] == 'h' else 1)

        grids.append(grid)
        label = lig_atom.tolist()
        label.append(0 if lig_atom_type_list[i] == 'h' else 1)
        labels.append(label)

    return grids, labels


def main():
    generate(32, 32, 7, 10)


if __name__ == '__main__':
    main()



