# Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import netket as nk

# sieć hipersześcianów o boku L i wymiarze d
# 1D Periodic Lattice
# lenght - It must always be >=1, but if pbc==True then the minimal valid length is 3.
g = nk.graph.Hypercube(length=12, n_dim=1, pbc=True)

# Boson Hilbert Space
# Przestrzeń Hilberta złożona ze stanów bozonowych.
# g - graficzna reprezentacja użyta do skonstruowania przestrzeni hilberta
# n_max: Maximum occupation for a site.//maksymalna liczba obsadzeń w punkcie sieci
# n_bosons: Constraint for the number of bosons. / ograniczenie liczby bozonów
hi = nk.hilbert.Boson(graph=g, n_max=3, n_bosons=9)

# Bose Hubbard Hamiltonian
# U - współczynnik oddziaływania Hubbarda
# V - potencjał zależyny od siatki (the hopping term)
# mu - potencjał chemiczny
ha = nk.operator.BoseHubbard(U=4.0, hilbert=hi, V=2.0)

# Jastrow Machine with Symmetry
# A fully connected Restricted Boltzmann Machine with lattice symmetries. This type of RBM has spin 1/2 hidden units
# (w pełni połączona maszyna Boltzmana z symetrią siatek)
# alpha- Hidden unit density.
# use_visible_bias
'''
hilbert – Hilbert space object for the system.
alpha – Hidden unit density.
use_visible_bias – If True then there would be a bias on the visible units. Default True.
use_hidden_bias – If True then there would be a bias on the visible units. Default True.'''
ma = nk.machine.RbmSpinSymm(alpha=20, hilbert=hi)
ma.init_random_parameters(seed=1234, sigma=0.01)

# Sampler
sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=ha)


# Stochastic gradient descent optimization
op = nk.optimizer.RmsProp(learning_rate=0.001)

# Variational Monte Carlo
# Variational Monte Carlo Optimisation
'''Finally, NetKet supports ground state searches for a given
many-body quantum Hamiltonian Ĥ. In this context, the task is to
optimize the parameters of a variational wavefunction Ψ in order
to minimize the energy ⟨ Ĥ ⟩ . The variational.Vmc driver class
contains the main logic to optimize a variational wavefunction
given a Hamiltonian, a sampler, and an optimizer
'''
vmc = nk.variational.Vmc(
    hamiltonian=ha,
    sampler=sa,
    optimizer=op,
    n_samples=1000,
    diag_shift=5e-3,
    use_iterative=False,
    method="Sr",
)

'''Alternatively, often more stable convergence can be 
achieved by using the stochastic reconfiguration (SR) method 
[41, 42], which approximates the imaginary time evolution of 
the system on the submanifold of variational states. The SR 
approach is closely related to the natural gradient descent
 method used in machine learning [43].'''

vmc.run(n_iter=100, output_prefix="test")



''' Zwraca energie układu'''