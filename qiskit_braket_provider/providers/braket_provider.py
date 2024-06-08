"""Amazon Braket provider."""

import warnings

from braket.aws import AwsDevice
from braket.circuits import Circuit as BraketCircuit
from braket.device_schema.dwave import DwaveDeviceCapabilities
from braket.device_schema.quera import QueraDeviceCapabilities
from braket.device_schema.xanadu import XanaduDeviceCapabilities
from qiskit.circuit import Gate
from qiskit.providers import ProviderV1
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import BasicSwap

from .braket_backend import BraketAwsBackend, BraketLocalBackend


class BraketProvider(ProviderV1):
    """BraketProvider class for accessing Amazon Braket backends.

    Example:
        >>> provider = BraketProvider()
        >>> backends = provider.backends()
        >>> backends
        [BraketBackend[Aria 1],
         BraketBackend[Aria 2],
         BraketBackend[Aspen-M-3],
         BraketBackend[Forte 1],
         BraketBackend[Harmony],
         BraketBackend[Lucy],
         BraketBackend[SV1],
         BraketBackend[TN1],
         BraketBackend[dm1]]
    """

    def backends(self, name=None, **kwargs):
        if kwargs.get("local"):
            return [
                BraketLocalBackend(name="braket_sv"),
                BraketLocalBackend(name="braket_dm"),
            ]
        names = [name] if name else None
        devices = AwsDevice.get_devices(names=names, **kwargs)
        # filter by supported devices
        # gate models are only supported
        supported_devices = [
            d
            for d in devices
            if not isinstance(
                d.properties,
                (
                    DwaveDeviceCapabilities,
                    XanaduDeviceCapabilities,
                    QueraDeviceCapabilities,
                ),
            )
        ]
        return [
            BraketAwsBackend(
                device=device,
                provider=self,
                name=device.name,
                description=f"AWS Device: {device.provider_name} {device.name}.",
                online_date=device.properties.service.updatedAt,
                backend_version="2",
            )
            for device in supported_devices
        ]


class AWSBraketProvider(BraketProvider):
    """AWSBraketProvider class for accessing Amazon Braket backends."""

    def __init_subclass__(cls, **kwargs):
        """This throws a deprecation warning on subclassing."""
        warnings.warn(
            f"{cls.__name__} is deprecated.", DeprecationWarning, stacklevel=2
        )
        super().__init_subclass__(**kwargs)

    def __init__(self):
        """This throws a deprecation warning on initialization."""
        warnings.warn(
            f"{self.__class__.__name__} is deprecated. Use BraketProvider instead",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()

    def to_braket(self, circuit, backend, native=True, topology=True):
        """
        Convert a Qiskit circuit to a Braket circuit respecting backend restrictions.

        Args:
            circuit (QuantumCircuit): The input Qiskit circuit.
            backend (BraketBackend): The backend to which the circuit will be transpiled.
            native (bool): Whether to use only native gates of the backend.
            topology (bool): Whether to respect the device topology.

        Returns:
            BraketCircuit: The converted Braket circuit.
        """
        if native:
            # Adjust angles and gate types based on backend restrictions
            circuit = self.adjust_angles(circuit, backend)
            if topology:
                circuit = self.adjust_topology(circuit, backend)

        # Convert the adjusted Qiskit circuit to a Braket circuit
        braket_circuit = self.convert_to_braket(circuit)

        return braket_circuit

    def adjust_angles(self, circuit, backend):
        """
        Adjust angles in the Qiskit circuit based on backend angle restrictions.

        Args:
            circuit (QuantumCircuit): The input Qiskit circuit.
            backend (BraketBackend): The backend to which the circuit will be transpiled.

        Returns:
            QuantumCircuit: The adjusted Qiskit circuit.
        """
        if backend.name.startswith("rigetti"):
            # Adjust angles for Rigetti backends
            for instr, _, _ in circuit.data:
                if isinstance(instr, Gate) and instr.name == "rx":
                    # Adjust angles to supported values
                    theta = instr.params[0]
                    new_theta = self.adjust_angle(theta)
                    instr.params[0] = new_theta
        # Add more angle adjustments for other backends as needed
        return circuit

    def adjust_angle(self, theta):
        """
        Adjust the angle to be within the supported range of the backend.

        Args:
            theta (float): The original angle.

        Returns:
            float: The adjusted angle.
        """
        # Adjust the angle to supported values
        supported_angles = [0, -1 / 2, 1 / 2, -1, 1]  # Example for Rigetti backend
        closest_angle = min(supported_angles, key=lambda x: abs(x - theta))
        return closest_angle

    def adjust_topology(self, circuit, backend):
        """
        Adjust the Qiskit circuit to respect the device topology of the backend.

        Args:
            circuit (QuantumCircuit): The input Qiskit circuit.
            backend (BraketBackend): The backend to which the circuit will be transpiled.

        Returns:
            QuantumCircuit: The adjusted Qiskit circuit.
        """
        # Get the coupling map of the backend
        coupling_map = CouplingMap(backend.configuration().coupling_map)

        # Create a pass manager with the BasicSwap pass to adjust for the topology
        pass_manager = PassManager()
        pass_manager.append(BasicSwap(coupling_map))

        # Run the pass manager on the circuit
        adjusted_circuit = pass_manager.run(circuit)

        return adjusted_circuit

    def convert_to_braket(self, circuit):
        """
        Convert the Qiskit circuit to a Braket circuit.

        Args:
            circuit (QuantumCircuit): The input Qiskit circuit.

        Returns:
            BraketCircuit: The converted Braket circuit.
        """
        braket_circuit = BraketCircuit()

        for instr, qargs, _ in circuit.data:
            if isinstance(instr, Gate):
                if instr.name == "h":
                    braket_circuit.h(qargs[0].index)
                elif instr.name == "x":
                    braket_circuit.x(qargs[0].index)
                elif instr.name == "y":
                    braket_circuit.y(qargs[0].index)
                elif instr.name == "z":
                    braket_circuit.z(qargs[0].index)
                elif instr.name == "s":
                    braket_circuit.s(qargs[0].index)
                elif instr.name == "sdg":
                    braket_circuit.si(qargs[0].index)
                elif instr.name == "t":
                    braket_circuit.t(qargs[0].index)
                elif instr.name == "tdg":
                    braket_circuit.ti(qargs[0].index)
                elif instr.name == "rx":
                    theta = instr.params[0]
                    braket_circuit.rx(theta, qargs[0].index)
                elif instr.name == "ry":
                    theta = instr.params[0]
                    braket_circuit.ry(theta, qargs[0].index)
                elif instr.name == "rz":
                    theta = instr.params[0]
                    braket_circuit.rz(theta, qargs[0].index)
                elif instr.name == "u1":
                    lambda_ = instr.params[0]
                    braket_circuit.phaseshift(lambda_, qargs[0].index)
                elif instr.name == "u2":
                    phi, lambda_ = instr.params
                    braket_circuit.u2(phi, lambda_, qargs[0].index)
                elif instr.name == "u3":
                    theta, phi, lambda_ = instr.params
                    braket_circuit.u3(theta, phi, lambda_, qargs[0].index)
                elif instr.name == "cx":
                    control, target = qargs
                    braket_circuit.cnot(control.index, target.index)
                elif instr.name == "cy":
                    control, target = qargs
                    braket_circuit.cy(control.index, target.index)
                elif instr.name == "cz":
                    control, target = qargs
                    braket_circuit.cz(control.index, target.index)
                elif instr.name == "ch":
                    control, target = qargs
                    braket_circuit.ch(control.index, target.index)
                elif instr.name == "crx":
                    theta = instr.params[0]
                    control, target = qargs
                    braket_circuit.crx(theta, control.index, target.index)
                elif instr.name == "cry":
                    theta = instr.params[0]
                    control, target = qargs
                    braket_circuit.cry(theta, control.index, target.index)
                elif instr.name == "crz":
                    theta = instr.params[0]
                    control, target = qargs
                    braket_circuit.crz(theta, control.index, target.index)
                elif instr.name == "cp":
                    lambda_ = instr.params[0]
                    control, target = qargs
                    braket_circuit.cphaseshift(lambda_, control.index, target.index)
                elif instr.name == "cu1":
                    lambda_ = instr.params[0]
                    control, target = qargs
                    braket_circuit.cphaseshift(lambda_, control.index, target.index)
                elif instr.name == "cu3":
                    theta, phi, lambda_ = instr.params
                    control, target = qargs
                    braket_circuit.cu3(theta, phi, lambda_, control.index, target.index)
                elif instr.name == "swap":
                    q0, q1 = qargs
                    braket_circuit.swap(q0.index, q1.index)
                elif instr.name == "ccx":
                    control1, control2, target = qargs
                    braket_circuit.ccnot(control1.index, control2.index, target.index)
                else:
                    raise ValueError(f"Unsupported gate: {instr.name}")

        return braket_circuit
