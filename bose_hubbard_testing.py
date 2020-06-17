import netket as nk


def RBM_bose_hubbard_model(U, J, particle_numbers, sites_number, number_hn_per_vn, iteration_number):
    g = nk.graph.Hypercube(length=sites_number, n_dim=1, pbc=True)

    hi = nk.hilbert.Boson(graph=g, n_max=particle_numbers - 1, n_bosons=particle_numbers)

    ha = nk.operator.BoseHubbard(U=U, J=J, mu=0., V=0., hilbert=hi)

    ma = nk.machine.RbmSpinSymm(alpha=number_hn_per_vn, hilbert=hi)
    ma.init_random_parameters(seed=1234, sigma=0.01)

    sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=ha)

    op = nk.optimizer.Sgd(learning_rate=0.05)

    vmc = nk.variational.Vmc(
        hamiltonian=ha,
        sampler=sa,
        optimizer=op,
        n_samples=1000,
        diag_shift=5e-3,
        use_iterative=False,
        method="Sr",
    )

    vmc.run(n_iter=iteration_number,
            output_prefix=f"architektury/testing_RBM/testing_hidden_neurons_number/test_J{J}_N{particle_numbers}_L{sites_number}_alpha_{number_hn_per_vn}_iter_{iteration_number}")


if __name__ == "__main__":
    Jj = [0.0]
    Uu = 1.0
    sites_n = 50
    iteration_n = 100
    particle_numbers = [50]
    hidden_neurons_number = [750]
    for j in Jj:
        for particle_n in particle_numbers:
            for hidden_n in hidden_neurons_number:
                alpha = int(round(hidden_n / sites_n))
                # if particle_n / sites_n > 1 and j == 0.29:
                #     j = 0.17

                RBM_bose_hubbard_model(Uu, j, particle_n, sites_n, alpha, iteration_n)
