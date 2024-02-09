# QHack2024IRL

My submission to QHack2024 IRL Hackathon.

In their 2010 paper, Cramer et. al. suggest that state tomography can be done efficiently (less than exponential timing in the system size) using unitary control and local measurements if the state in question is a Matrix Product State.

Cramer et. al. provide an algorithm for approximating an arbitrary state with a matrix product state which may be used in conjunction with standard quantum state tomography. They call this a Modified Singular Value Thresholding algorithm.

We implement this algorithm as well as basic state tomography in this notebook and show how they may be used together.

QST uses a complete set of observables’ measurements to fully characterize the quantum state. This includes the lengthy classical task of reconstructing the state from the individual measurement outcomes.

Here we implement single-qubit state tomography in the Pauli basis. We define observables using phases, which we repeat for p values of r. The more phase offsets that we use in the procedure, the more accurate our tomography will be. For each phase we measure with respect to each operator in our operator basis of Pauli matrices. The Pauli basis is tomographically complete. The Pauli operators span the entire space from which we are sampling. We measure X, Y and Z by collapsing the system to an eigenstate of each Hamiltonian. In practice we are limited in most quantum computers to Z-axis measurements so we need to perform rotations of the state that map the eigenstates of our Pauli operators onto the computational basis states, which are the eigenstates of the Z operator. The rotation needed for X-axis measurement is the Hadamard gate. The rotation required for the Y-axis measurement is S dagger followed by a Hadamard. The results of these measurements are expectations which may be used to reconstruct the quantum state.

The goal of our modified singular value thresholding (Cramer et. al., 2010) is a low-rank approximation, |\psi>, of the state. This problem fits nicely in the classical singular value thresholding (Cai, Candès, Shen, 2010) set-up.

We see the recognizable Bloch vector corresponding to the state X |0> in an example. We also see that the density
matrix |1><1| is approximated well by a Matrix Product State via the Modified SVT algorithm. This means that we would be
able to efficiently decompose the state into a product of Ai operators. This is appealing since these
Ai could be used to prepare the state in question again for future experiments.

This generalizes to states that are well approximated by MPSs. However this sort of tomography is less efficient when it
comes to states that are not well approximated by MPSs. In these cases, traditional state tomography will be slow and it is the best we can do.

Please copy the cells where we estimate the MPS and graph the Bloch vectors to try out the algorithm with more of states we have prepared!
