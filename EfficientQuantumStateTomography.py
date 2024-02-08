import pennylane as qml
from pennylane.operation import Operation
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import partial


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
            qml.measure(qubit)
            return qml.probs(qubit)


        results.append(x_circuit(phase, qubit))
        results.append(y_circuit(phase, qubit))
        results.append(z_circuit(phase, qubit))

    bloch_vectors = _analyze_tomographic_results(qubit, exp_vector, results)

    return results, bloch_vectors


def _analyze_tomographic_results(qubit, exp_vector, results):
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


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    def state_prep(theta, phi, delta, wires):
        qml.U3(theta, phi, delta, wires=wires)

    for t in [-1, 0, 1]:
        for p in [-1, 0, 1]:
            for d in [-1, 0, 1]:

                print("Tomography result for state:")
                print(np.matmul(qml.U3(t, p, d, wires=0).matrix(), np.array([[0], [1]])))

                res, vecs = tomography(state_prep=partial(state_prep, t, p, d, 0), qubit=0, phases=10)

                vecs = np.array([[0, 0, 0] + vec for vec in vecs])  # start vectors from the origin for plotting

                X, Y, Z, U, V, W = zip(*vecs)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.quiver(X, Y, Z, U, V, W)
                ax.set_xlim([-1, 1])
                ax.set_ylim([-1, 1])
                ax.set_zlim([-1, 1])
                plt.show()
