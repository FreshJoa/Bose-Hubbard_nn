from bose_hubbard_testing import RBM_bose_hubbard_model
import json
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

FILES_DIRECTORY_PATH = 'architektury/phase_diagram_recostruction'
J = [0.0001, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.28, 0.29]
U = 1.
lattice_size = [25, 20, 15, 10]
iterations = 200
alpha = 3


def linear_function(x, a, b):
    return a * x + b




def get_chemical_potencial(j):
    chemical_potential_plus = []
    chemical_potential_minus = []
    one_over_lattice = []
    for lattice in lattice_size:
        data_plus = json.load(
            open(f"{FILES_DIRECTORY_PATH}/phase_diagram_J{j}_N{lattice + 1}_L{lattice}_alpha{alpha}_iter_{iterations}.log"))
        data_minus = json.load(
            open(f"{FILES_DIRECTORY_PATH}/phase_diagram_J{j}_N{lattice - 1}_L{lattice}_alpha{alpha}_iter_{iterations}.log"))
        data = json.load(
            open(f"{FILES_DIRECTORY_PATH}/phase_diagram_J{j}_N{lattice}_L{lattice}_alpha{alpha}_iter_{iterations}.log"))

        chemical_potential_plus.append(data_plus['Output'][-1]['Energy']['Mean'] - data['Output'][-1]['Energy']['Mean'])
        chemical_potential_minus.append(
            data['Output'][-1]['Energy']['Mean']-data_minus['Output'][-1]['Energy']['Mean'])
        one_over_lattice.append(1 / lattice)

    popt_plus, pcov_plus = curve_fit(linear_function, one_over_lattice, chemical_potential_plus)
    popt_minus, pcov_minus = curve_fit(linear_function, one_over_lattice, chemical_potential_minus)

    plt.figure(figsize=(15, 10))
    plt.plot(one_over_lattice, chemical_potential_plus, label='u+')
    plt.plot(one_over_lattice, chemical_potential_minus, label='u-')
    plt.plot(one_over_lattice, [x*popt_plus[0]+popt_plus[1] for x in one_over_lattice], label='u+ fittetd')
    plt.plot(one_over_lattice, [x*popt_minus[0]+popt_minus[1] for x in one_over_lattice], label='u- fittetd')
    plt.legend()
    plt.title(f'j = {j}')
    plt.show()


    return popt_plus[1], popt_minus[1]


def plot_phase_diagram():
    chemical_potential_plus = []
    chemical_potential_minus = []
    for j in J:
        chemical_potentials = get_chemical_potencial(j)
        chemical_potential_plus.append(chemical_potentials[0])
        chemical_potential_minus.append(chemical_potentials[1])

    plt.figure(figsize=(15, 10))
    plt.plot(chemical_potential_plus, J, label='u+')
    plt.plot(chemical_potential_minus, J, label='u-')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_phase_diagram()
    # for j in J:
    #     for lattice in lattice_size:
    #         RBM_bose_hubbard_model(U, j, lattice, lattice, alpha, iterations,
    #                                'architektury/phase_diagram_recostruction')
    #         RBM_bose_hubbard_model(U, j, lattice - 1, lattice, alpha, iterations,
    #                                'architektury/phase_diagram_recostruction')
    #         RBM_bose_hubbard_model(U, j, lattice + 1, lattice, alpha, iterations,
    #                                'architektury/phase_diagram_recostruction')
