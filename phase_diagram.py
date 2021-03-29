from bose_hubbard_testing import RBM_bose_hubbard_model
import json
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import statistics

# J = [1e-07, 0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.17, 0.20, 0.23, 0.25, 0.27, 0.30, 0.31, 0.32, 0.33, 0.35, 0.37, 0.40]
J=[ 0.30, 0.31, 0.32, 0.33, 0.35, 0.37, 0.40]
# J= [0.28]
U = 1.
lattice_size = { 20:2000}

alpha = 5

FILES_DIRECTORY_PATH = "DWARF/phase_tansition_BH"

OUTPUT_DIRECTORY ='DWARF/phase_tansition_BH'

PREFIX = 'Sgd_lr_0.05_'
SHOW_PLOTS = False


def linear_function(x, a, b):
    return a * x + b


def reject_outliers_2(data, m=1.2):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 0.5)
    indexes = np.where(s < m)[0]
    data = np.array(data).take(indexes)
    return data.tolist(), indexes


def get_chemical_potencial(j):
    chemical_potential_plus = []
    chemical_potential_minus = []
    one_over_lattice = []
    data_plus_list = []
    data_minus_list = []
    data_list = []
    variance_data_list = []
    variance_data_list_plus = []
    variance_data_list_minus = []
    for lattice in lattice_size:
        try:
            data_plus = json.load(
                open(
                    f'{FILES_DIRECTORY_PATH}/phase_diagram_J{j}_N{lattice +1}_L{lattice}_alpha{alpha}_iter_{iterations}_{PREFIX}.log'))
            data_minus = json.load(
                open(
                    f'{FILES_DIRECTORY_PATH}/phase_diagram_J{j}_N{lattice - 1}_L{lattice}_alpha{alpha}_iter_{iterations}_{PREFIX}.log'))
            data = json.load(
                open(
                    f'{FILES_DIRECTORY_PATH}/phase_diagram_J{j}_N{lattice}_L{lattice}_alpha{alpha}_iter_{iterations}_{PREFIX}.log'))
        except:
            continue
        max_iteration = min(min(len(data_plus['Output']), len(data_minus['Output'])), len(data['Output']))
        print(max_iteration)
        for iteration in range(0, max_iteration, 1):
            data_plus_list.append(data_plus['Output'][iteration]['Energy']['Mean'])
            variance_data_list_plus.append(data_plus['Output'][iteration]['Energy']['Sigma'])
            data_minus_list.append(data_minus['Output'][iteration]['Energy']['Mean'])
            variance_data_list_minus.append(data_minus['Output'][iteration]['Energy']['Sigma'])
            data_list.append(data['Output'][iteration]['Energy']['Mean'])
            variance_data_list.append(data['Output'][iteration]['Energy']['Sigma'])

        # plus_indexes = np.argsort(variance_data_list_plus)[:100]
        # plus_value = np.mean([data_plus_list[x] for x in plus_indexes])
        #
        # minus_indexes = list(np.argsort(variance_data_list_minus))[:100]
        # minus_value = np.mean([data_minus_list[x] for x in minus_indexes])
        #
        # indexes = list(np.argsort(variance_data_list))[:100]
        # value = np.mean([data_list[x] for x in indexes]
        print(j)
        start_index = round(0.3 * max_iteration)
        print(start_index)
        plus_value = np.mean(data_plus_list[start_index:])
        minus_value = np.mean(data_minus_list[start_index:])
        value = np.mean(data_list[start_index:])

        chemical_potential_plus.append(plus_value - value)
        chemical_potential_minus.append(value - minus_value)
        one_over_lattice.append(1 / lattice)
    #
    # cleaned_chemical_potential_plus, indexes_plus = reject_outliers_2(chemical_potential_plus)
    # cleaned_chemical_potential_minus, indexes_minus = reject_outliers_2(chemical_potential_minus)
    # one_over_lattice_plus = np.array(one_over_lattice).take(indexes_plus).tolist()
    # one_over_lattice_minus = np.array(one_over_lattice).take(indexes_minus).tolist()
    cleaned_chemical_potential_plus = chemical_potential_plus
    cleaned_chemical_potential_minus = chemical_potential_minus
    one_over_lattice_plus = one_over_lattice
    one_over_lattice_minus = one_over_lattice

    popt_plus, pcov_plus = curve_fit(linear_function, one_over_lattice_plus, cleaned_chemical_potential_plus)
    popt_minus, pcov_minus = curve_fit(linear_function, one_over_lattice_minus, cleaned_chemical_potential_minus)
    plt.figure(figsize=(6, 4))

    plt.scatter(one_over_lattice_plus, cleaned_chemical_potential_plus, label='u+', color='#FFCE04')
    plt.scatter(one_over_lattice_minus, cleaned_chemical_potential_minus, label='u-', color='#9F5CAC')
    one_over_lattice.append(0)
    plt.plot(one_over_lattice, [x * popt_plus[0] + popt_plus[1] for x in one_over_lattice], label='u+ dopasowane',
             color='#FFCE04')
    plt.plot(one_over_lattice, [x * popt_minus[0] + popt_minus[1] for x in one_over_lattice], label='u- dopasowane',
             color='#9F5CAC')
    plt.xlabel('1/L')
    plt.ylabel('u+-')
    plt.grid()
    plt.legend()
    plt.title(f'J = {j}')
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.savefig(f'{OUTPUT_DIRECTORY}/seminariumu_1_L_j{j}_lattice_{lattice_size[0]}_{lattice_size[-1]}.png')
    plt.close()
    # return cleaned_chemical_potential_plus[-1], cleaned_chemical_potential_minus[-1]
    return popt_plus[1], popt_minus[1]


def plot_phase_diagram():
    chemical_potential_plus = []
    chemical_potential_minus = []
    for j in J:
        chemical_potentials = get_chemical_potencial(j)
        chemical_potential_plus.append(chemical_potentials[0])
        chemical_potential_minus.append(chemical_potentials[1])

    plt.figure(figsize=(5, 4))
    plt.scatter(J, chemical_potential_plus, label='u+', color='#FFCE04')
    plt.scatter(J, chemical_potential_minus, label='u-', color='#9F5CAC')
    plt.plot([0.29, 0.29], [-0.2, 0.8], color='#91908C')
    plt.legend()
    plt.ylabel('u/U')
    plt.xlabel('J/U')
    plt.grid()
    # plt.xticks([round(i, 2) for i in J])
    # plt.title(f'Phase diagram for rho=1, for lattice sizes from {lattice_size[0]} to {lattice_size[-1]}')
    plt.title(f'Diagram fazowy dla ρ=1')
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.savefig(
            f'{OUTPUT_DIRECTORY}/diagram_fazowy_rho1_j_{J[0]}_{J[-1]}_lattice_{lattice_size[0]}_{lattice_size[-1]}.png')
    plt.close()


def run_calculations():
    for j in J:
        for lattice in lattice_size:
            RBM_bose_hubbard_model(U, j, lattice, lattice, alpha, iterations,
                                   f'{FILES_DIRECTORY_PATH}/phase_diagram_J{j}_N{lattice}_L{lattice}_alpha{alpha}_iter_{iterations}_{PREFIX}')
            RBM_bose_hubbard_model(U, j, lattice - 1, lattice, alpha, iterations,
                                   f'{FILES_DIRECTORY_PATH}/phase_diagram_J{j}_N{lattice - 1}_L{lattice}_alpha{alpha}_iter_{iterations}_{PREFIX}')
            RBM_bose_hubbard_model(U, j, lattice + 1, lattice, alpha, iterations,
                                   f'{FILES_DIRECTORY_PATH}/phase_diagram_J{j}_N{lattice + 1}_L{lattice}_alpha{alpha}_iter_{iterations}_{PREFIX}')


def get_energy_plots():
    for j in J:
        for lattice in lattice_size:
            data_plus_list = []
            data_minus_list = []
            data_list = []
            variance_data_plus_list = []
            variance_data_minus_list = []
            variance_data_list = []

            data_plus = json.load(
                open(
                    f'{FILES_DIRECTORY_PATH}/phase_diagram_J{j}_N{lattice + 1}_L{lattice}_alpha{alpha}_iter_{iterations}_{PREFIX}.log'))
            data_minus = json.load(
                open(
                    f'{FILES_DIRECTORY_PATH}/phase_diagram_J{j}_N{lattice - 1}_L{lattice}_alpha{alpha}_iter_{iterations}_{PREFIX}.log'))
            data = json.load(
                open(
                    f'{FILES_DIRECTORY_PATH}/phase_diagram_J{j}_N{lattice}_L{lattice}_alpha{alpha}_iter_{iterations}_{PREFIX}.log'))

            plus_len = len(data_plus['Output'])
            minus_len = len(data_minus['Output'])
            data_len = len(data['Output'])
            max_iteration = min(data_len, min(plus_len, minus_len))

            for iteration in range(0, max_iteration, 1):
                data_plus_list.append(data_plus['Output'][iteration]['Energy']['Mean'])
                variance_data_plus_list.append((data_plus['Output'][iteration]['EnergyVariance']['Mean']))
                data_minus_list.append(data_minus['Output'][iteration]['Energy']['Mean'])
                variance_data_minus_list.append((data_minus['Output'][iteration]['EnergyVariance']['Mean']))
                data_list.append(data['Output'][iteration]['Energy']['Mean'])
                variance_data_list.append((data['Output'][iteration]['EnergyVariance']['Mean']))
            plt.figure(figsize=(10, 7))
            plt.scatter(range(0, max_iteration, 1), data_list[0:max_iteration],
                        label=f'E(L, N=L)', color='#9F5CAC', s=3)
            plt.scatter(range(0, max_iteration, 1), data_plus_list[0:max_iteration],
                        label=f'E(L, N=L+1)', color='#FFCE04', s=3)
            plt.scatter(range(0, max_iteration, 1), data_minus_list[0:max_iteration],
                        label=f'E(L, N=L-1)', color='#91908C', s=3)
            # plt.ylim(-1.75, 0)
            plt.legend()
            plt.grid()
            plt.ylabel('Energia')
            plt.xlabel('iteracja')
            plt.title(f't={j}, liczba węzłów L ={lattice}')
            if SHOW_PLOTS:
                plt.show()
            else:
                plt.savefig(f'{OUTPUT_DIRECTORY}/energy_plot_j{j}_lattice_{lattice}.png')
            plt.close()


def get_energy_parameters_plots():
    data_types = ["N=L", "N=L+1", "N=L-1"]
    data = {}
    energy_parameters = defaultdict(list)
    prob_params = ["Mean", "Sigma", "Taucorr"]
    parameters = ["Energy", "EnergyVariance"]
    for j in J:
        for lattice in lattice_size:
            data[data_types[0]]= [json.loads(line) for line in
                open(
                    f'{FILES_DIRECTORY_PATH}/phase_diagram_J{j}_N{lattice}_L{lattice}_alpha{alpha}_iter_{iterations}_{PREFIX}.log')]
            data[data_types[1]]=(json.load(
                open(
                    f'{FILES_DIRECTORY_PATH}/phase_diagram_J{j}_N{lattice + 1}_L{lattice}_alpha{alpha}_iter_{iterations}_{PREFIX}.log')))
            data[data_types[2]]=(json.load(
                open(
                    f'{FILES_DIRECTORY_PATH}/phase_diagram_J{j}_N{lattice-1}_L{lattice}_alpha{alpha}_iter_{iterations}_{PREFIX}.log')))

        data_len = len(data[data_types[0]]['Output'])
        plus_len = len(data[data_types[1]]['Output'])
        minus_len = len(data[data_types[2]]['Output'])
        max_iteration = min(data_len, min(plus_len, minus_len))

        for data_type in data_types:
            for iteration in range(0, max_iteration, 1):
                for parameter in parameters:
                    for prob_param in prob_params:
                        energy_parameters[f'{data_type}.{parameter}.{prob_param}'].append(
                            data[data_type]['Output'][iteration][parameter][prob_param])
                energy_parameters[f"{data_type}.Acceptance"].append(data[data_type]['Output'][iteration]["Acceptance"][0])

        for parameter in parameters:
            for prob_param in prob_params:
                plt.figure(figsize=(10, 7))
                for data_type in data_types:

                    plt.scatter(range(0, max_iteration, 1), energy_parameters[f'{data_type}.{parameter}.{prob_param}'],
                                label=f'{data_type}', s=3)
                plt.grid()
                plt.legend()
                plt.ylabel('')
                plt.xlabel('iteracja')
                plt.title(f'{parameter}: {prob_param}')
                if SHOW_PLOTS:
                    plt.show()
                else:
                    plt.savefig(f'{OUTPUT_DIRECTORY}/energy_prameters_j{j}_lattice_{lattice}_{parameter}_{prob_param}.png')
                plt.close()

        plt.figure(figsize=(10, 7))
        for data_type in data_types:

            plt.scatter(range(0, max_iteration, 1), energy_parameters[f'{data_type}.Acceptance'],
                        label=f'{data_type}', s=3)
        plt.grid()
        plt.ylabel('')
        plt.legend()

        plt.xlabel('iteracja')
        plt.title(f'Acceptance')
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.savefig(f'{OUTPUT_DIRECTORY}/energy_prameters_j{j}_lattice_{lattice}_Acceptance.png')
        plt.close()


def get_energy_plot_for_many_J():
    start= 500
    data_types = ["N=L", "N=L+1", "N=L-1"]
    data = {}
    prob_param = "Mean"
    parameters = ["Energy", "EnergyVariance"]
    mediana_u = {}
    for j in J:

        for lattice, iterations in lattice_size.items():
            # if j in [0.3, 0.31, 0.32, 0.33, 0.35]:
            #     iterations = 8000
            data[data_types[0]]= json.load(
                open(f'{FILES_DIRECTORY_PATH}/{format(j, ".2f")}/phase_diagram_J{j}_N{lattice}_L{lattice}_alpha{alpha}_iter_{iterations}_{PREFIX}.log'))
            data[data_types[1]]=(json.load(
                open(
                    f'{FILES_DIRECTORY_PATH}/{format(j, ".2f")}/phase_diagram_J{j}_N{lattice + 1}_L{lattice}_alpha{alpha}_iter_{iterations}_{PREFIX}.log')))
            data[data_types[2]]=(json.load(
                open(
                    f'{FILES_DIRECTORY_PATH}/{format(j, ".2f")}/phase_diagram_J{j}_N{lattice-1}_L{lattice}_alpha{alpha}_iter_{iterations}_{PREFIX}.log')))

            data_len = len(data[data_types[0]]['Output'])
            plus_len = len(data[data_types[1]]['Output'])
            minus_len = len(data[data_types[2]]['Output'])
            max_iteration = min(data_len, min(plus_len, minus_len))
            # max_iteration=1000
            energy_parameters = defaultdict(list)

            for data_type in data_types:
                    for iteration in range(0, max_iteration, 1):
                        for parameter in parameters:
                            energy_parameters[f'{data_type}.{parameter}.{prob_param}'].append(
                                data[data_type]['Output'][iteration][parameter][prob_param])

            plt.figure(figsize=(10, 7))
            for data_type in data_types:
                plt.scatter(range(start, max_iteration, 1), energy_parameters[f'{data_type}.{"Energy"}.{prob_param}'][start:max_iteration],
                            label=f'lattice = {lattice}, {data_type}', s=5)
                mediana_u[data_type] = statistics.median(energy_parameters[f'{data_type}.{"Energy"}.{prob_param}'][start:max_iteration])

                # plt.plot(range(start, max_iteration, 1), energy_parameters[f'{data_type}.{"EnergyVariance"}.{prob_param}'][start:max_iteration])
            delta = (mediana_u[data_types[1]] - mediana_u[data_types[0]]) - (mediana_u[data_types[0]] - mediana_u[data_types[2]])
            plt.grid()
            plt.legend()
            plt.ylabel('')
            plt.xlabel('iteracja')
            plt.title(f'J = {j}, delta = {delta}')
            if SHOW_PLOTS:
                plt.show()
            else:
                plt.savefig(f'{OUTPUT_DIRECTORY}/{format(j, ".2f")}/energy_plot_j{j}_lattice_{lattice}_{PREFIX}.png')
            plt.close()


def plot_delta_J():
    delta = []
    for j in J:
        chemical_potentials = get_chemical_potencial(j)
        delta.append(chemical_potentials[0] - chemical_potentials[1])
        if j == 0.29:
            delta_j029 = delta[-1]
    plt.figure(figsize=(5, 4))
    plt.grid()
    plt.scatter(J, delta, color='#9F5CAC')
    plt.plot([0.29, 0.29], [0, 1], color='#FFCE04')
    plt.plot([0, 0.5], [delta_j029, delta_j029], color='#FFCE04')
    plt.xlabel('t/U')
    plt.ylabel('Δ = (u+) - (u-)')
    # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, 0.52, 0.02))
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.savefig(f'{OUTPUT_DIRECTORY}/delta_J_prezentacja.png')
    plt.close()


if __name__ == "__main__":
    # print(FILES_DIRECTORY_PATH)

    # plot_phase_diagram()
    # run_calculations()
    get_energy_plot_for_many_J()
    # plot_delta_J()
    # get_energy_parameters_plots()