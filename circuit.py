import cirq
from math import *

class QuantamLayer:
    def __init__(self):
        pass


class Kernel:
    def __init__(self):
        self.weight = [255,255,255,255]
    
    def update_weight(self, w):
        self.weight = w
    
    def kernel1(self, P):
        # detects edge for weight (255,255,255)
        Q = [cirq.GridQubit(i,0) for i in range(4)]
        keys = ["q0", "q1", "q2", "q3"]

        circuit = cirq.Circuit()
        for i in range(4):
            circuit.append(cirq.ry((P[i] + self.weight[i])/255 *pi).on(Q[i]))

        for i in range(3):
            circuit.append(cirq.CNOT(Q[i],Q[i+1]))
        
        for i in range(4):
            circuit.append(cirq.measure(Q[i], key=keys[i]))
        
        return circuit, keys

    def kernel2(self, P):
        # 2x2 kernel for image compresion with restoration
        Q = [cirq.GridQubit(i,0) for i in range(4)]
        W = [cirq.GridQubit(i,1) for i in range(3)]
        keys = ["q0", "q1", "q2", "q3"]

        circuit = cirq.Circuit()
        # for i in range(4):
        #     circuit.append(cirq.H(Q[i]))

        for i in range(4):
            circuit.append(cirq.ry(P[i]/255 * pi).on(Q[i]))
        
        for i in range(3):
            circuit.append(cirq.rz(self.weight[i]/255 * pi).on(W[i]))
        
        for i in range(3):
            circuit.append(cirq.TOFFOLI(W[i], Q[i], Q[i+1]))

        for i in range(3):
            circuit.append(cirq.ZZ(Q[i], Q[i+1]))
        
        for i in range(4):
            circuit.append(cirq.measure(Q[i], key=keys[i]))
        return circuit, keys


if __name__ == '__main__':
    def main(circuit, keys):
        # circuit = simple_quantum_circuit()
        print("Quantum Circuit:")
        print(circuit)

        # Use a simulator to run the quantum circuit
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1000)

        # Print the measurement results
        print("\nMeasurement Results:")
        # print(result, type(result))
        for k in keys:
            print(k, ": ",result.histogram(key=k), (result.histogram(key=k)[1]))

    
    circuit, keys = Kernel().kernel2([0,12,230,0])
    main(circuit, keys)

