"""Tests for QuantumCircuit and Gate."""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quantum_devops.circuit import QuantumCircuit, Gate


class TestGate:
    def test_valid_gate_creation(self):
        g = Gate("H", (0,))
        assert g.name == "H"
        assert g.qubits == (0,)
        assert g.params == ()

    def test_gate_with_params(self):
        import math
        g = Gate("RZ", (1,), (math.pi / 4,))
        assert g.params == (math.pi / 4,)

    def test_unsupported_gate_raises(self):
        with pytest.raises(ValueError, match="Unsupported gate"):
            Gate("TOFFOLI", (0, 1, 2))

    def test_two_qubit_detection(self):
        assert Gate("CNOT", (0, 1)).is_two_qubit
        assert not Gate("H", (0,)).is_two_qubit

    def test_parametric_detection(self):
        assert Gate("RZ", (0,), (1.0,)).is_parametric
        assert not Gate("H", (0,)).is_parametric

    def test_repr(self):
        g = Gate("H", (0,))
        assert "H" in repr(g)
        assert "q[0]" in repr(g)


class TestQuantumCircuit:
    def test_basic_creation(self):
        qc = QuantumCircuit(3, name="test")
        assert qc.num_qubits == 3
        assert qc.name == "test"
        assert len(qc.gates) == 0

    def test_zero_qubits_raises(self):
        with pytest.raises(ValueError):
            QuantumCircuit(0)

    def test_h_gate(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        assert len(qc.gates) == 1
        assert qc.gates[0].name == "H"

    def test_cnot_gate(self):
        qc = QuantumCircuit(2)
        qc.cnot(0, 1)
        assert qc.gates[0].name == "CNOT"
        assert qc.two_qubit_gate_count == 1

    def test_cnot_same_qubit_raises(self):
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError, match="control and target must differ"):
            qc.cnot(0, 0)

    def test_rz_gate(self):
        import math
        qc = QuantumCircuit(1)
        qc.rz(0, math.pi / 4)
        assert qc.gates[0].name == "RZ"
        assert abs(qc.gates[0].params[0] - math.pi / 4) < 1e-10

    def test_rx_gate(self):
        import math
        qc = QuantumCircuit(1)
        qc.rx(0, math.pi / 2)
        assert qc.gates[0].name == "RX"

    def test_measure_single(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.measure(0)
        assert qc.measurement_count == 1

    def test_measure_all(self):
        qc = QuantumCircuit(3)
        qc.measure()
        assert qc.measurement_count == 3

    def test_out_of_range_qubit_raises(self):
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError, match="out of range"):
            qc.h(5)

    def test_gate_count_excludes_measurements(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cnot(0, 1)
        qc.measure()
        assert qc.gate_count == 2
        assert qc.measurement_count == 2

    def test_depth_empty(self):
        qc = QuantumCircuit(2)
        assert qc.depth == 0

    def test_depth_sequential(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(0, 1.0)
        assert qc.depth == 2

    def test_depth_parallel(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)  # Can run in parallel with h(0)
        # Both should be depth 1 since they use different qubits
        assert qc.depth == 1

    def test_depth_cnot(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cnot(0, 1)
        assert qc.depth == 2

    def test_chaining(self):
        qc = QuantumCircuit(2)
        result = qc.h(0).cnot(0, 1).measure()
        assert result is qc
        assert qc.gate_count == 2

    def test_validate_valid_circuit(self):
        qc = QuantumCircuit(2)
        qc.h(0).cnot(0, 1).measure()
        issues = qc.validate()
        assert issues == []

    def test_validate_no_measurements(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        issues = qc.validate()
        assert any("No measurements" in i for i in issues)

    def test_validate_empty_circuit(self):
        qc = QuantumCircuit(1)
        issues = qc.validate()
        assert any("empty" in i for i in issues)

    def test_validate_gate_after_measurement(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure(0)
        qc._add_gate(Gate("H", (0,)))  # Manually add gate after measure
        issues = qc.validate()
        assert any("after measurement" in i for i in issues)

    def test_draw(self):
        qc = QuantumCircuit(2, name="Bell")
        qc.h(0).cnot(0, 1).measure()
        drawing = qc.draw()
        assert "Bell" in drawing
        assert "H" in drawing
        assert "CNOT" in drawing

    def test_repr(self):
        qc = QuantumCircuit(2, name="test")
        qc.h(0).measure()
        r = repr(qc)
        assert "test" in r
        assert "qubits=2" in r
