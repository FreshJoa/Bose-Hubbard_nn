import matplotlib.pyplot as plt
import json
from collections import defaultdict
import itertools
import numpy as np

# iters = defaultdict(list)
# energy = defaultdict(list)
# sigma = defaultdict(list)
# evar = defaultdict(list)
# data_names = ['RBM_Sym_alpha_20_AdaMax', 'Momentum_beta_05', 'RBM_Sym_alpha_20_AdaDelta', 'RBM_Sym_alpha_20_AdaGard',
#               'RBM_Sym_alpha_20_Momentum', 'RBM_Sym_alpha_20_SGD_0.1',
#               'RBM_Sym_alpha_20_SGD_0.05', 'RmsProp_learning_rate_0.02']
#
# for name in data_names:
#     data = json.load(open(f"architektury/RBM_Sym_optimizer/{name}/test.log"))
#     for iteration in data["Output"]:
#         iters[name].append(iteration["Iteration"])
#         energy[name].append(iteration["Energy"]["Mean"])
#         sigma[name].append(iteration["Energy"]["Sigma"])
#         evar[name].append(iteration["EnergyVariance"]["Mean"])
#
# plt.figure()
# plt.ylabel("Energy")
# plt.xlabel("Iterations")
# plt.errorbar(iters[data_names[0]], energy[data_names[0]], yerr=sigma[data_names[0]], color="#F70A79", linestyle='-', linewidth=1)
# plt.errorbar(iters[data_names[1]], energy[data_names[1]], yerr=sigma[data_names[1]], color="#930BEE", linestyle='-', linewidth=1)
# plt.errorbar(iters[data_names[2]], energy[data_names[2]], yerr=sigma[data_names[2]], color="#0B1AEE", linestyle='-', linewidth=1)
# plt.errorbar(iters[data_names[3]], energy[data_names[3]], yerr=sigma[data_names[3]], color="#0BB9EE", linestyle='-', linewidth=1)
# plt.errorbar(iters[data_names[4]], energy[data_names[4]], yerr=sigma[data_names[4]], color="#07D51C", linestyle='-', linewidth=1)
# plt.errorbar(iters[data_names[5]], energy[data_names[5]], yerr=sigma[data_names[5]], color="#F7E11A", linestyle='-', linewidth=1)
# plt.errorbar(iters[data_names[6]], energy[data_names[6]], yerr=sigma[data_names[6]], color="#F77A1A", linestyle='-', linewidth=1)
# plt.errorbar(iters[data_names[7]], energy[data_names[7]], yerr=sigma[data_names[7]], color="#950A0A", linestyle='-', linewidth=1)
#
# # plt.xticks(np.arange(0, 100, 5.0))
# plt.rcParams['figure.figsize'] = (8, 6)
# plt.legend(data_names)
# plt.title('Optimizer testing')
# plt.savefig('energy_plot_optimizers.png')
# plt.show()
# exact = 0
#
# Uu = 1.0
# particle_n = 6
# iteration_n = 100
# hidden_neurons_number = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# sites_n = 50
#
# J = 0.29
# iters = defaultdict(list)
# energy = defaultdict(list)
# sigma = defaultdict(list)
# evar = defaultdict(list)
#
# for particle_n in hidden_neurons_number:
#     if particle_n / sites_n > 1 and J == 0.29:
#         J = 0.17
#     alpha = int(round(100 / sites_n))
#     data = json.load(
#         open(
#             f"architektury/testing_RBM/testing_particle_number/test_J{J}_N{particle_n}_L{sites_n}_alpha_{alpha}_iter_{iteration_n}.log.log"))
#     for iteration in data["Output"]:
#         iters[particle_n].append(iteration["Iteration"])
#         energy[particle_n].append(iteration["Energy"]["Mean"])
#         sigma[particle_n].append(iteration["Energy"]["Sigma"])
#         evar[particle_n].append(iteration["EnergyVariance"]["Mean"])
#
# plt.figure()
# plt.ylabel("Energy")
# plt.xlabel("Iterations")
# plt.errorbar(iters[hidden_neurons_number[0]], energy[hidden_neurons_number[0]], yerr=sigma[hidden_neurons_number[0]], color="#F70A79",
#              linestyle='-', linewidth=1)
# plt.errorbar(iters[hidden_neurons_number[1]], energy[hidden_neurons_number[1]], yerr=sigma[hidden_neurons_number[1]], color="#930BEE",
#              linestyle='-', linewidth=1)
# plt.errorbar(iters[hidden_neurons_number[2]], energy[hidden_neurons_number[2]], yerr=sigma[hidden_neurons_number[2]], color="#0B1AEE",
#              linestyle='-', linewidth=1)
# plt.errorbar(iters[hidden_neurons_number[3]], energy[hidden_neurons_number[3]], yerr=sigma[hidden_neurons_number[3]], color="#0BB9EE",
#              linestyle='-', linewidth=1)
# plt.errorbar(iters[hidden_neurons_number[4]], energy[hidden_neurons_number[4]], yerr=sigma[hidden_neurons_number[4]], color="#07D51C",
#              linestyle='-', linewidth=1)
# plt.errorbar(iters[hidden_neurons_number[5]], energy[hidden_neurons_number[5]], yerr=sigma[hidden_neurons_number[5]], color="#F7E11A",
#              linestyle='-', linewidth=1)
# plt.errorbar(iters[hidden_neurons_number[6]], energy[hidden_neurons_number[6]], yerr=sigma[hidden_neurons_number[6]], color="#F77A1A",
#              linestyle='-', linewidth=1)
# plt.errorbar(iters[hidden_neurons_number[7]], energy[hidden_neurons_number[7]], yerr=sigma[hidden_neurons_number[7]], color="#950A0A",
#              linestyle='-', linewidth=1)
# plt.errorbar(iters[hidden_neurons_number[8]], energy[hidden_neurons_number[8]], yerr=sigma[hidden_neurons_number[8]], color="#E8CFF0",
#              linestyle='-', linewidth=1)
# plt.errorbar(iters[hidden_neurons_number[9]], energy[hidden_neurons_number[9]], yerr=sigma[hidden_neurons_number[9]], color="#1A8C1E",
#              linestyle='-', linewidth=1)
# plt.errorbar(iters[hidden_neurons_number[10]], energy[hidden_neurons_number[10]], yerr=sigma[hidden_neurons_number[10]],
#              color="#542A54",
#              linestyle='-', linewidth=1)
# # plt.axhline(y=exact, linewidth=2, color="k")
#
# # plt.xticks(np.arange(0, 100, 5.0))
# plt.rcParams['figure.figsize'] = (20, 15)
# plt.legend(hidden_neurons_number)
# plt.title(
#     f'Testing number of particles for U=1, dla q<1 J=0.29, dla q>1 J={J}, \n number of sites = {sites_n},\n number of hidden neurons = 100')
# plt.savefig(f'architektury/testing_RBM/testing_particle_number/energy_plot_particle_number_J={J}.png')
# plt.show()

# exact = 0
#
# Uu = 1.0
# particle_n = 25
# iteration_n = 100
# hidden_neurons_number = [ 50, 100, 200, 300, 500, 750, 1000]
# sites_n = 50
#
# J = 0.0
# iters = defaultdict(list)
# energy = defaultdict(list)
# sigma = defaultdict(list)
# evar = defaultdict(list)
#
# for hnn in hidden_neurons_number:
#     # if particle_n / sites_n > 1 and J == 0.29:
#     #     J = 0.17
#     print(hnn)
#     alpha = int(round(hnn / sites_n))
#     data = json.load(
#         open(
#             f"architektury/testing_RBM/testing_hidden_neurons_number/test_J{J}_N{particle_n}_L{sites_n}_alpha_{alpha}_iter_{iteration_n}.log"))
#     for iteration in data["Output"]:
#         iters[hnn].append(iteration["Iteration"])
#         energy[hnn].append(iteration["Energy"]["Mean"])
#         sigma[hnn].append(iteration["Energy"]["Sigma"])
#         evar[hnn].append(iteration["EnergyVariance"]["Mean"])
#
# plt.figure(figsize=(10, 7))
# plt.ylabel("Energy")
# plt.xlabel("Iterations")
# plt.errorbar(iters[hidden_neurons_number[0]], energy[hidden_neurons_number[0]], yerr=sigma[hidden_neurons_number[0]],
#              color="#C70E2A",
#              linestyle='-', linewidth=1)
# plt.errorbar(iters[hidden_neurons_number[1]], energy[hidden_neurons_number[1]], yerr=sigma[hidden_neurons_number[1]],
#              color="#FAA111",
#              linestyle='-', linewidth=1)
# plt.errorbar(iters[hidden_neurons_number[2]], energy[hidden_neurons_number[2]], yerr=sigma[hidden_neurons_number[2]],
#              color="#7BC712",
#              linestyle='-', linewidth=1)
# plt.errorbar(iters[hidden_neurons_number[3]], energy[hidden_neurons_number[3]], yerr=sigma[hidden_neurons_number[3]],
#              color="#E773EB",
#              linestyle='-', linewidth=1)
# plt.errorbar(iters[hidden_neurons_number[4]], energy[hidden_neurons_number[4]], yerr=sigma[hidden_neurons_number[4]],
#              color="#11A9FA",
#              linestyle='-', linewidth=1)
# plt.errorbar(iters[hidden_neurons_number[5]], energy[hidden_neurons_number[5]], yerr=sigma[hidden_neurons_number[5]],
#              color="#8F00FC",
#              linestyle='-', linewidth=1)
# plt.errorbar(iters[hidden_neurons_number[6]], energy[hidden_neurons_number[6]], yerr=sigma[hidden_neurons_number[6]],
#              color="#F74F16",
#              linestyle='-', linewidth=1)
# # plt.errorbar(iters[hidden_neurons_number[7]], energy[hidden_neurons_number[7]], yerr=sigma[hidden_neurons_number[7]], color="#950A0A"#1A8C1E,
# #              linestyle='-', linewidth=1)
# # plt.errorbar(iters[hidden_neurons_number[8]], energy[hidden_neurons_number[8]], yerr=sigma[hidden_neurons_number[8]], color="#E8CFF0",
# #              linestyle='-', linewidth=1)
# # plt.errorbar(iters[hidden_neurons_number[9]], energy[hidden_neurons_number[9]], yerr=sigma[hidden_neurons_number[9]], color="#1A8C1E",
# #              linestyle='-', linewidth=1)
# # plt.errorbar(iters[hidden_neurons_number[10]], energy[hidden_neurons_number[10]], yerr=sigma[hidden_neurons_number[10]],
# #              color="#542A54",
# #              linestyle='-', linewidth=1)
# # plt.axhline(y=exact, linewidth=2, color="k")
#
# # plt.xticks(np.arange(0, 100, 5.0))
# plt.legend(hidden_neurons_number)
# plt.grid()
# plt.xlim(0, 100)
#
# plt.title(
#     f'Testing number of hidden neurons for U=1, J = {J} , number of sites = {sites_n},\n number of particles = {particle_n}')
# plt.savefig(f'architektury/testing_RBM/testing_hidden_neurons_number/energy_plot_N_H_particle_n={particle_n}_J={J}.png')
# plt.show()

rho = [0.5, 1, 1.5, 2]
n_h = [200, 200, 200, 200]

plt.figure(figsize=(10, 4))
plt.scatter(rho, n_h, color='#FAA111')
plt.xlabel('N/L')
plt.xticks(np.arange(0.5, 2.05, 0.5))
plt.yticks(np.arange(190, 211, 10))
plt.ylabel('Number of hidden neurons')
plt.savefig(f'architektury/testing_RBM/testing_hidden_neurons_number/N_h_NL.png')
plt.show()