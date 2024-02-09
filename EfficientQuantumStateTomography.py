import pennylane as qml
from pennylane.operation import Operation
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import partial, reduce
from qutip.visualization import hinton
from sympy import symbols, Eq, solve


dev = qml.device('default.qubit', wires=2, shots=1000)


class SDG(Operation):
    num_params = 0
    num_wires = 1
    par_domain = "R"

    @property
    def matrix(self):
        return np.array(
            [
                [1, 0],
                [0, -complex(0, 1)]
            ]
        )

    def decomposition(self):
        return [qml.U1(3 * np.pi / 2, wires=self.wires)]


def tomography(state_prep, qubit, phases):
    """
    Replicates the current circuit and performs measurements in each of the three orthogonal axes of the Bloch
    sphere to determine the qubit's state.

    :param state_prep: The state prep to inject into each of our tomography circuits.
    :param qubit: The target qubit.
    :param phases: The number of relative phases to measure with respect to each axis.
    that it has been split up over time.
    :return: The results from each circuit.
    """

    results = []
    bloch_vectors = []

    exp_vector = range(0, phases)
    index = 0

    while index < len(exp_vector):
        phase = 2 * np.pi * index / (len(exp_vector) - 1)
        index += 1

        @qml.qnode(dev)
        def x_circuit(phase, qubit):
            state_prep()
            qml.U1(phase, wires=qubit)
            qml.Hadamard(wires=qubit)
            qml.measure(qubit)
            return qml.probs(qubit)

        @qml.qnode(dev)
        def y_circuit(phase, qubit):
            state_prep()
            qml.U1(phase, wires=qubit)
            SDG(wires=qubit)
            qml.Hadamard(wires=qubit)
            qml.measure(qubit)
            return qml.probs(qubit)

        @qml.qnode(dev)
        def z_circuit(phase, qubit):
            state_prep()
            qml.U1(phase, wires=qubit)
            qml.measure(qubit)  # pennylane lets us measure other things!
            return qml.probs(qubit)


        results.append(x_circuit(phase, qubit))
        results.append(y_circuit(phase, qubit))
        results.append(z_circuit(phase, qubit))

    bloch_vectors = analyze_tomographic_results(qubit, exp_vector, results)

    return results, bloch_vectors


def estimate_measured_operator(bloch_vectors):
    """
    Estimates the operator measured by tomography.
    :param bloch_vectors: The reconstructed Bloch vectors.
    :return: The operator estimate.
    """

    def estimate_one(bv):
        return (1 / 2) * (qml.PauliX(wires=0).matrix() * bv[0]  # 1 / 2^N for N qubits
                          + qml.PauliY(wires=0).matrix() * bv[1]
                          + qml.PauliZ(wires=0).matrix() * bv[2])

    op = np.array([[np.mean([estimate_one(bv)[0][0] for bv in bloch_vectors]), np.mean([estimate_one(bv)[0][1] for bv in bloch_vectors])],
                    [np.mean([estimate_one(bv)[1][0] for bv in bloch_vectors]), np.mean([estimate_one(bv)[1][1] for bv in bloch_vectors])]])

    return op


def estimate_density_matrix(combined_results, operator_basis):
    """
    Estimates the measured density matrix from a set of reconstructed Bloch vectors.
    :param combined_results: The combined tomographic results (probabilities associated with each outcome).
    :param operator_basis: The operator basis used for state tomography.
    :return: The measured state.
    """

    projs = []
    vals = []

    # build A
    for op in operator_basis:

        # get the projectors

        # first we get the eigenvectors
        vs, vecs = np.linalg.eigh(op)
        for v in vs:
            vals.append(v)

        # now we map them to projectors
        for vec in vecs:

            # map to a 2D array
            vec = np.reshape(vec, (-1, 1))

            # build projector
            proj = np.matmul(vec, np.conjugate(np.transpose(vec)))
            projs.append(proj)

    A = np.array([[proj] for proj in projs])

    # map our eigenvalues to z eigenvalues TODO: don't hard-code this
    for i in range(len(vals)):
        if i % 2 == 0:
            vals[i] = 1
        else:
            vals[i] = 0

    # average together our data
    acc = combined_results[0]
    for results in combined_results[1:]:
        for j, op in enumerate(results):
            for k in list(op.keys()):
                acc[j][k] = acc[j][k] + results[j][k]

    for l, op in enumerate(acc):
        for m in list(op.keys()):
            acc[l][m] = acc[l][m] / len(combined_results)

    # now we build p from our observations
    p = []

    for i in range(len(operator_basis)):
        op_outcomes = acc[i]
        p.append([op_outcomes[str(int(vals[i * 2]))]])
        p.append([op_outcomes[str(int(vals[i * 2 + 1]))]])

    p = np.array(p)

    # now we solve for rho by linear inversion
    Adagger = np.conjugate(np.transpose(A))

    # this will be invertible if basis is complete
    inv = np.linalg.inv(np.matmul(Adagger, A))

    rho = np.matmul(np.matmul(inv, Adagger), p)

    return rho



def combine_tomographic_results(qubit, exp_vector, results, operator_basis):
    """
    Combines our results into probabilities for each outcome.
    :param qubit: The qubit we do tomography on.
    :param exp_vector: An array of increments to rotate our state by.
    :param results: The results of our tomographic circuits.
    :param operator_basis: The operator basis used for tomography.
    :return: The probabilities associated with each measurement projector.
    """
    probs = []
    for exp_index in exp_vector:
        bloch = [0, 0, 0]
        for bloch_index in range(len(operator_basis)):
            p_zero = 0
            p_one = 0
            circuit_index = 3 * exp_index + bloch_index
            data = results[circuit_index]
            for readout in range(len(data)):
                format_str = '{' + f'0:0{int(len(data) / 2)}b' + '}'
                qubit_readout = format_str.format(readout)[qubit]
                if qubit_readout == '0':
                    p_zero += data[readout]
                elif qubit_readout == '1':
                    p_one += data[readout]
            bloch[bloch_index] = {'0': p_zero, '1': p_one}
        probs.append(bloch)

    return probs


def analyze_tomographic_results(qubit, exp_vector, results):
    """
    Create Bloch vectors from the probability distributions gained from our measurements in the
    various bases.
    :param qubit: The qubit to do tomography on.
    :param exp_vector: An array of increments to rotate our state by.
    :param results: The results from running our tomography circuits.
    :return: The Bloch vectors constructed from the data.
    """
    bloch_vector = ['x', 'y', 'z']
    bloch_vectors = []
    for exp_index in exp_vector:
        bloch = [0, 0, 0]
        for bloch_index in range(len(bloch_vector)):
            p_zero = 0
            p_one = 0
            circuit_index = 3 * exp_index + bloch_index
            data = results[circuit_index]
            for readout in range(len(data)):
                format_str = '{' + f'0:0{int(len(data) / 2)}b' + '}'
                qubit_readout = format_str.format(readout)[qubit]
                if qubit_readout == '0':
                    p_zero += data[readout]
                elif qubit_readout == '1':
                    p_one += data[readout]
            bloch[bloch_index] = p_zero - p_one
        bloch_vectors.append(bloch)

    print("Observed Bloch vectors:")
    print(bloch_vectors)

    return bloch_vectors


def modified_singular_value_thresholding(density_matrix, operator_basis, steps, deltas):
    """
    Implements the modified singular value thresholding algorithm detailed in Cramer et. al.
    Builds a Matrix Product State approximation of the given density matrix, in the provided operator basis.

    WARNING: this method should only be used with states that are Matrix Product States (MPSs).

    :param density_matrix: The density matrix we want to approximate with an MPS.
    :param operator_basis: The operator basis we want to use to build our MPS.
    :param steps: The number of steps to allow for converging on the solution.
    :param deltas: The multipliers of each step's estimates as weighted in the final solution.
    :return:
    """

    assert steps == len(deltas)

    #our initial guess
    Yn = np.array([[1., 0.],
                   [0., 0.]])

    # our optimization steps
    for i in range(steps):

        # find our eigenvalues and eigenvectors
        eigen_values, eigen_vectors = np.linalg.eigh(Yn)

        # find the largest eigenvalue and associated eigenvector
        yval = max(eigen_values)
        yn = eigen_vectors[list(eigen_values).index(yval)]

        # create an MPS approximation
        Xn = reduce(lambda x, y: x + y,
                    [(1 / 2) * yval * np.matmul(np.matmul(np.conjugate(np.transpose(yn)), basis_op), yn)
                     for basis_op in operator_basis])  # (1 / 2^N) for N qubits

        # update our overall estimate
        Yn = Yn + deltas[i] * (density_matrix - Xn)

    return Yn


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    # our tomography operator basis
    pauli_basis = [qml.PauliX(wires=0).matrix(), qml.PauliY(wires=0).matrix(), qml.PauliY(wires=0).matrix()]

    # our state preparations
    state_preps = []

    # simple state preparations
    state_preps.append(lambda: qml.PauliX(wires=0))
    state_preps.append(lambda: qml.PauliY(wires=0))
    state_preps.append(lambda: qml.PauliZ(wires=0))

    # random state preparations
    def state_prep(theta, phi, delta, wires):
        return qml.U3(theta, phi, delta, wires=wires)

    for t in [-1, 0, 1]:
        for p in [-5, 0, 5]:
            for d in [-5, 0, 5]:
                state_preps.append(partial(state_prep, t, p, d, 0))

    # execute tomography
    for prep in state_preps:
        print("Tomography result for state:")
        print(np.matmul(prep().matrix(), np.array([[1], [0]])))

        res, vecs = tomography(state_prep=prep, qubit=0, phases=10)

        combined = combine_tomographic_results(qubit=0, exp_vector=range(0, 10), results=res, operator_basis=pauli_basis)

        density_matrix = estimate_density_matrix(combined, pauli_basis)
        print("Estimated density matrix:")
        print(density_matrix)

        # graph density matrix
        fig, ax = hinton(density_matrix)
        plt.show()

        # graph Bloch vectors
        vecs = np.array([[0, 0, 0] + vec for vec in vecs])  # start vectors from the origin for plotting

        X, Y, Z, U, V, W = zip(*vecs)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(X, Y, Z, U, V, W)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        plt.show()

        # closest MPS to our density matrix
        steps = 10
        svt_matrix = modified_singular_value_thresholding(density_matrix, pauli_basis, steps, [0.1 for k in range(steps)])

        print(f"The closest Matrix Product State to our matrix estimated with {steps} steps:")
        print(svt_matrix)
