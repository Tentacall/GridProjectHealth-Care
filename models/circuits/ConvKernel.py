# imports
import cirq
from math import pi

def kernel2(P, weight):
    Q = [cirq.GridQubit(i,0) for i in range(4)]
    W = [cirq.GridQubit(i,1) for i in range(3)]
    keys = ["q0", "q1", "q2", "q3"]

    circuit = cirq.Circuit()
    # for i in range(4):
    #     circuit.append(cirq.H(Q[i]))

    for i in range(4):
        circuit.append(cirq.ry(P[i] * pi).on(Q[i]))
    
    for i in range(3):
        circuit.append(cirq.rx(weight[i] * pi).on(W[i]))
    
    for i in range(3):
        circuit.append(cirq.TOFFOLI(W[i], Q[i], Q[i+1]))

    for i in range(3):
        circuit.append(cirq.ZZ(Q[i], Q[i+1]))
    
    for i in range(4):
        circuit.append(cirq.measure(Q[i], key=keys[i]))
    return circuit, keys

def kernel3(P, weight):
    circuit = cirq.Circuit()
    # bias addition kernel
    Q = [cirq.GridQubit(i,0) for i in range(4)]
    W = cirq.GridQubit(0,1)
    keys = ["q"]
    for i in range(4):
        circuit.append(cirq.ry(P[i] * pi).on(Q[i]))
    
    for i in range(4):
        circuit.append(cirq.rx(weight[i]*pi).on(W))
        circuit.append(cirq.CNOT(W, Q[i]))

    circuit.append(cirq.measure(Q[3], key=keys[i]))
    
    return circuit, keys


def main(circuit, keys):
    # circuit = simple_quantum_circuit()
    print("Quantum Circuit:")
    print(circuit)

    # Use a simulator to run the quantum circuit
    # simulator = cirq.Simulator()
    simulator = qsimcirq.QSimSimulator()
    result = simulator.run(circuit, repetitions=1000)

    # Print the measurement results
    print("\nMeasurement Results:")
    # print(result, type(result))
    for k in keys:
      print(k, ": ",result.histogram(key=k), (result.histogram(key=k)[1]))


if __name__ == '__main__':
    import qsimcirq
    circuit, keys = kernel3([0.0, 0.0, 0.0, 0.0], [0.1, -0.111612, 0.20, 0.1])
    main(circuit, keys)