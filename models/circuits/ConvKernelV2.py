from math import pi
import cirq

def kernel4(P, weight):
    '''
(0, 0): ───Ry(0)──────@────────────────────────────────────────────────────────────
                      │
(0, 1): ───Rx(0.1π)───@───Rx(-0.112π)───@───Rx(0.2π)───@───Rx(0.1π)───@────────────
                      │                 │              │              │
(1, 0): ───Ry(0)──────┼─────────────────@──────────────┼──────────────┼────────────
                      │                 │              │              │
(1, 1): ──────────────X─────────────────X──────────────X──────────────X───M('q')───
                                                       │              │
(2, 0): ───Ry(0)───────────────────────────────────────@──────────────┼────────────
                                                                      │
(3, 0): ───Ry(0)──────────────────────────────────────────────────────@────────────
    '''
    circuit = cirq.Circuit()
    # bias addition kernel
    Q = [cirq.GridQubit(i,0) for i in range(4)]
    W = cirq.GridQubit(0,1)
    R = cirq.GridQubit(1,1)
    keys = ["q"]
    for i in range(4):
        circuit.append(cirq.ry(P[i] * pi).on(Q[i]))
    
    for i in range(4):
        circuit.append(cirq.rx(weight[i]*pi).on(W))
        circuit.append(cirq.CCNOT(W, Q[i], R))

    circuit.append(cirq.measure(R, key=keys[0]))
    
    return circuit, keys

if __name__ == '__main__':
    import qsimcirq
    from ConvKernel import main
    circuit, keys = kernel4([0.0, 0.0, 0.0, 0.0], [0.1, -0.111612, 0.20, 0.1])
    main(circuit, keys)