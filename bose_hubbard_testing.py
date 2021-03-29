import netket as nk

def RBM_bose_hubbard_model(U, J, particle_numbers, sites_number, alpha, iteration_number, output_directory):
    g = nk.graph.Hypercube(length=sites_number, n_dim=1, pbc=True)

    hi = nk.hilbert.Boson(graph=g, n_max=particle_numbers - 1, n_bosons=particle_numbers)

    ha = nk.operator.BoseHubbard(U=U, J=J, mu=0., V=0., hilbert=hi)

    ma = nk.machine.RbmSpinSymm(alpha=alpha, hilbert=hi)
    ma.init_random_parameters(seed=1234, sigma=0.01)

    sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=ha)

    # op = nk.optimizer.AdaMax(alpha=0.01, beta1=0.9, beta2=0.99, epscut=10 ** (-8))
    op = nk.optimizer.Sgd(learning_rate=0.05)
    # The optimizer object that determines how the VMC wavefunction is optimized.
    # op = nk.optimizer.AdaMax(alpha=0.02, beta1=0.01, beta2=0.999)

    vmc = nk.variational.Vmc(
        hamiltonian=ha,
        sampler=sa,
        optimizer=op,
        n_samples=1000,  # Number of Markov Chain Monte Carlo sweeps to be performed at each step of the optimization.
        diag_shift=5e-3,  # The regularization parameter in stochastic reconfiguration. The default is 0.01.
        use_iterative=True,  # Whether to use the iterative solver in the Sr method (this is extremely useful
        # when the number of parameters to optimize is very large). The default is false.
        method="Sr",  # The chosen method to learn the parameters of the wave-function.
        # The default is Sr (stochastic reconfiguration).
    )

    vmc.run(n_iter=iteration_number,
            output_prefix=output_directory)


if __name__ == "__main__":
    J = [0.000001]
    U = 1.
    lattice_size = [12]
    iterations = 40
    alpha = 4

    # FILES_DIRECTORY_PATH = f'architektury/phase_diagram_recostruction/J_{J[0]}_{J[-1]}_lattice_{lattice_size[0]}_{lattice_size[-1]}_iter{iterations}'
    FILES_DIRECTORY_PATH = '.'

    PREFIX = 'Sgd_lr_0.05_'

    RBM_bose_hubbard_model(U, J[0], lattice_size[0], lattice_size[0], alpha, iterations, 'test')
