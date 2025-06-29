'''
In this file, we define the classes 'VQE_ansatz' (representing an ansatz for the VQE quantum circuit) and 
'Evolutionary_Algorithm' handling the logic of the genetic algorithm.
The VQE_ansatz class is equipped with the relevant methods of the algorithm, such as crossover, mutation, fitness, etc.
'''

import numpy as np
import pennylane as qml
import random

class VQE_ansatz():

    '''
    The class VQE_ansatz defines the individuals of the genetic algorithm: the quantum circuits that are
    used for VQE. The class consists of a circuit initialisation function as well as various methods that
    allow for the evaluation, pairing and mutation of the circuits in the context of the EA.
    '''
    ########## CLASS PARAMETERS ############
    
    # Define a dtype for each gate (this is a class attribute common for all instances of the class)
    gate_dtype = np.dtype([
        ("name", "U10"),  # e.g. 'RX', 'CNOT'
        ("qubit_id", "i4"),  # single qubit this gate is applied on
        ("affected_qubits", "O"),  # list of affected qubits (for multi-qubit gates)
        ("parameters", "O"),  # list or float (e.g., angle for rotation gates)
        ("control_qubits", "O"),  # list of control qubits
        ("target_qubits", "O")  # list of target qubits
    ])

    # We define the finite gate set to consist of identity, discrete rotations by certain angles, CNOT and Hadamard
    gate_set_names=["Id", "RX", "RY", "RZ", "CNOT", "H"]

    angles_x=np.array([0.53, 1.57, 2.36]) # Angles (in radians) for the X rotations
    angles_y=np.array([0.79, 1.65, 2.64]) # Angles (in radians) for the Y rotations
    angles_z=np.array([0.21, 1.18, 2.93]) # Angles (in radians) for the Z rotations

    angles={"RX": angles_x, "RY": angles_y, "RZ": angles_z}

    ################ INSTANTIATION OF THE CLASS #################

    def __init__(self, num_qubits, depth, hamiltonian_label, h=None):
        '''
        Initializes a class instance (quantum circuit) with a given number of qubits and depth
        as well as Hamiltonian as observable.
        '''
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit_matrix = np.empty((num_qubits, depth), dtype=self.gate_dtype)
        self.hamiltonian = None
        self.hamiltonian_label = hamiltonian_label
        self.energy = np.inf # initialize infinite energy since the EA is trying to minimize energy
        if h is not None:
            self.h = h # Transverse field coefficient (only used if the Hamiltonian is transverse field Ising)
        self.magnetization_observable = None
        self.z_magnetization = None
        # Generate the circuit matrix with random gates, build the corresponding quantum circuit, and evaluate its energy
        self.build_hamiltonian()    
        self.generate_random_circuit_matrix()
        self.tensor_to_circuit()
        self.fitness()

    ################ METHODS #################

    def generate_random_circuit_matrix(self):
        '''
        Generates a random circuit matrix, as needed for initial initialization of circuits in 
        the evolutionary algorithm.
        '''
        for d in range(0, self.depth): # We first iterate over the depth
            i = 0
            while i < self.num_qubits:
                gate = random.choice(self.gate_set_names) # We randomly select one element

                # To avoid index problems, the last qubit "cannot have" a CNOT gate (it can, but via the previous qubit)
                if i == self.num_qubits-1:
                    gate = random.choice([element for element in self.gate_set_names if element != "CNOT"])
                
                if gate =="CNOT":
                    # Select control and targets, which are two consecutive qubits, simulating limited qubit connectivity
                    list = [i, i+1]
                    random_control = random.choice(list)
                    random_target = list[1] if random_control == list[0] else list[0]
                    # Encode the CNOT in the position (i+1, d) of the circuit matrix [(i,d) is skipped]
                    self.circuit_matrix[i+1, d] = (gate, random_target, [i,i+1], [], [random_control], [random_target])
                    i += 1 # We skip the following qubit as it is already included in the CNOT gate

                elif gate=="RX" or gate=="RY" or gate=="RZ":
                    random_angle=random.choice(self.angles[gate])
                    self.circuit_matrix[i, d] = (gate, i, [i], [random_angle], [], [i])
                
                else: # "H" or "Id"
                    self.circuit_matrix[i, d] = (gate, i, [i], [], [], [i]) 
                
                i += 1
                
    def build_hamiltonian(self):
        '''
        This method constructs the indicated Hamiltonian, which is used as observable
        for measurement (and thus fitness evaluation).
        '''
        coef = []
        ops = []

        if self.hamiltonian_label == 'ising':
            # Easiest version of the Ising model (all couplings have the same strength and no transverse field)
            for i in range(self.num_qubits - 1):
                ops.append(qml.PauliZ(i)@qml.PauliZ(i+1))
                coef.append(-1)

        elif self.hamiltonian_label == 'heisenberg':
            for i in range(self.num_qubits - 1):
                ops.append(qml.PauliX(i)@qml.PauliX(i+1) + qml.PauliY(i)@qml.PauliY(i+1) + qml.PauliZ(i)@qml.PauliZ(i+1))
                coef.append(-1)

        elif self.hamiltonian_label == 'magnetization':
            for i in range(self.num_qubits):
                ops.append(qml.PauliZ(i))
                coef.append(1/self.num_qubits)

        elif self.hamiltonian_label == 'transverse_field_ising':
            # ZZ interaction terms
            for i in range(self.num_qubits - 1):
                ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
                coef.append(-1)

            # X transverse field terms
            for i in range(self.num_qubits):
                ops.append(qml.PauliX(i))
                coef.append(self.h)
    
        self.hamiltonian = qml.Hamiltonian(coef, ops)
    
    def build_magnetization(self):
        '''
        This method constructs the magnetization, which is used to study the phase transition
        in the transverse field Ising model. We define it separately from the Hamiltonian to avoid 
        having to re-define the Hamiltonian every time we want to measure the magnetization.
        '''
        coef = []
        ops = []

        for i in range(self.num_qubits):
            ops.append(qml.PauliZ(i))
            coef.append(1/self.num_qubits) # for normalization
    
        self.magnetization_observable = qml.Hamiltonian(coef, ops)

    def tensor_to_circuit(self, magnetization=False):
        '''
        This method transforms a given tensor (i.e., circuit encoding in a generalized matrix) to 
        a PennyLane quantum circuit that can be executed in the process of evaluating fitness.
        '''
        
        dev = qml.device("default.qubit", wires=self.num_qubits)

        def apply_gate(gate):
            name, qubit_id, affected_qubits, parameters, control_qubits, target_qubits = gate
            if name == "RX":
                qml.RX(parameters[0], wires=qubit_id)
            elif name == "RY":
                qml.RY(parameters[0], wires=qubit_id)
            elif name == "RZ":
                qml.RZ(parameters[0], wires=qubit_id)
            elif name == "CNOT":
                qml.CNOT(wires=control_qubits + target_qubits)
            elif name=="Id":
                qml.Identity(wires=qubit_id)
            elif name=="H":
                qml.Hadamard(wires=qubit_id)


        def circuit():
            for depth_idx in range(self.circuit_matrix.shape[1]): # iterating over all layers
                for q in range(self.circuit_matrix.shape[0]): # iterating over all qubits (in one layer)
                    gate = self.circuit_matrix[q, depth_idx]  
                    if gate["name"] != "":
                        apply_gate(gate)
            if magnetization:
                return qml.expval(self.magnetization_observable)
            
            return qml.expval(self.hamiltonian)  # VQE observable
        
        self.qnode = qml.QNode(circuit, dev) # This is the quantum node that executes the circuit
        
        
    def fitness(self):  
        '''
        This method returns the fitness of a quantum circuit. Note that evaluating the fitness function
        relies on the tensor_to_circuit() method, which implements the observable measurement.
        '''   
        self.energy = self.qnode()
        return self.energy
    
    def visualize_circuit(self):
        '''
        This method serves to visualize a circuit.
        '''
        print(qml.draw(self.qnode)())
        
    def crossover(self, partner, proba_entanglement_aware):
        '''
        This method defines the crossover operation between two quantum circuits.
        It accepts a quantum circuit and a partner and performs a pre-defined crossover logic
        to recombine features of the circuits. It then returns the children circuits as new
        quantum circuits in the EA.

        proba_entanglement_aware: float between 0 and 1
            Probability of performing entanglement-aware crossover.
        '''

        if random.uniform(0, 1) > proba_entanglement_aware:
            # Here we perform a 'blind' crossover, i.e., we randomly select a crossover point
            crossover_point = random.randint(1, self.depth - 1)

        else:    
            # Detect entangled blocks in both parents
            parent1_blocks = self.detect_entangled_blocks()
            parent2_blocks = partner.detect_entangled_blocks()
            
            # Find compatible crossover points using gate commutativity
            crossover_point = self.find_commuting_block_pair(parent1_blocks, parent2_blocks)
            
        # Create new children through block recombination
        child1_matrix = np.concatenate((
            self.circuit_matrix[:, :crossover_point],
            partner.circuit_matrix[:, crossover_point:]
        ), axis=1)
        
        child2_matrix = np.concatenate((
            partner.circuit_matrix[:, :crossover_point],
            self.circuit_matrix[:, crossover_point:]
        ), axis=1)

        # Initialize children with recombined circuits
        child1 = VQE_ansatz(self.num_qubits, self.depth, self.hamiltonian_label, self.h)
        child2 = VQE_ansatz(self.num_qubits, self.depth, self.hamiltonian_label, self.h)
        
        child1.circuit_matrix = child1_matrix
        child2.circuit_matrix = child2_matrix
        
        child1.tensor_to_circuit()
        child2.tensor_to_circuit()
        child1.fitness()
        child2.fitness()
        
        return child1, child2
        
    def detect_entangled_blocks(self):
        '''Identify entangled gate sequences using CNOT patterns'''
        blocks = []
        current_block = []
        
        for d in range(self.depth):
            layer_entanglement = set()
            for q in range(self.num_qubits):
                gate = self.circuit_matrix[q, d]
                if gate["name"] == "CNOT":
                    qubit_pair = tuple(sorted([gate["control_qubits"][0], gate["target_qubits"][0]]))
                    layer_entanglement.add(qubit_pair)
            
            if layer_entanglement:
                layer_entanglement = frozenset(layer_entanglement)
                if not current_block or any(pair in layer_entanglement for pair in current_block[-1]):
                    current_block.append(layer_entanglement)
                else:
                    blocks.append(current_block)
                    current_block = [layer_entanglement]
            else:
                if current_block:
                    blocks.append(current_block)
                    current_block = []
        
        return blocks

    def find_commuting_block_pair(self, blocks1, blocks2):
        '''Find compatible crossover points using commutativity rules'''
        max_score = -1
        best_point = self.depth // 2  # Default midpoint
        
        for i in range(1, len(blocks1)):
            for j in range(1, len(blocks2)):
                # Score based on shared entanglement patterns
                score = len(set(blocks1[i]).intersection(blocks2[j]))
                if score > max_score:
                    max_score = score
                    best_point = min(i, j)  # Take earliest compatible point
        
        return best_point

    def mutation(self, rate, TwoGate_mut_rate=0.1):
        '''
        This method defines a mutation of a circuit as a random alteration of given gates
        with a pre-defined rate. 
        TwoGate_mut_rate defines the rate of two qubit rotations
        '''
        if random.random() < rate:
            if random.random() > TwoGate_mut_rate: # Here we check a single qubit

                i = random.randint(0, self.num_qubits-1)
                d = random.randint(0, self.depth-1)
                
                if self.circuit_matrix[i, d][0] in ["RX", "RY", "RZ", "Id", "H"]: # In this case the qubit is affected by a single-qubit gate   
                    
                    gate = random.choice(["RX", "RY", "RZ", "Id", "H"])  # We randomly select a single qubit gate

                    if gate == "RX" or gate == "RY" or gate == "RZ": 
                        random_angle = random.choice(self.angles[gate])
                        self.circuit_matrix[i, d] = (gate, i, [i], [random_angle], [], [i])

                    else: # Gate is replaced by Id or H
                        self.circuit_matrix[i, d] =( gate, i, [i], [], [], [i])

                    # Rebuild the circuit and evaluate fitness
                    self.tensor_to_circuit()
                    self.fitness()

                elif self.circuit_matrix[i, d][0] == "CNOT":

                    control_index= self.circuit_matrix[i,d][4][0]
                    target_index = self.circuit_matrix[i,d][5][0]

                    gate1 = random.choice(["RX", "RY", "RZ", "Id", "H"])  # We randomly select a rotation for the first qubit
                    gate2 = random.choice(["RX", "RY", "RZ", "Id", "H"])  # We randomly select a rotation for the second qubit

                    # If the gate is a rotation, we randomly select an angle
                    if gate1 in ["RX", "RY", "RZ"]:
                        if gate2 in ["RX", "RY", "RZ"]:
                            random_angle1 = random.choice(self.angles[gate1])
                            random_angle2 = random.choice(self.angles[gate2])
                            self.circuit_matrix[control_index, d] = (gate1, control_index, [control_index], [random_angle1], [], [control_index])
                            self.circuit_matrix[target_index, d] = (gate2, target_index, [target_index], [random_angle2], [], [target_index])

                        else:
                            random_angle1=random.choice(self.angles[gate1])
                            self.circuit_matrix[control_index, d] = (gate1, control_index, [control_index], [random_angle1], [], [control_index])
                            self.circuit_matrix[target_index, d] = (gate2, target_index, [target_index], [], [], [target_index])

                    else:
                        if gate2 in ["RX", "RY", "RZ"]:
                            random_angle2=random.choice(self.angles[gate2])
                            self.circuit_matrix[control_index, d] = (gate1, control_index, [control_index], [], [], [control_index])
                            self.circuit_matrix[target_index, d] = (gate2, target_index, [target_index], [random_angle2], [], [target_index])

                        else:
                            self.circuit_matrix[control_index, d] = (gate1, control_index, [control_index], [], [], [control_index])
                            self.circuit_matrix[target_index, d] = (gate2, target_index, [target_index], [], [], [target_index])
                            
            else:
                # In this case we combine two consecutive single qubit gates to add a CNOT

                i = random.randint(0, self.num_qubits-1)
                d = random.randint(0, self.depth-1)

                while self.circuit_matrix[i, d][0] == "CNOT": # to avoid mutating a CNOT into a CNOT
                    i = random.randint(0, self.num_qubits-1)
                    d = random.randint(0, self.depth-1)

                if i == self.num_qubits-1: # i.e. if index is equal to num_qubits-1 we reduce it by 1 so to avoid index conflicts
                    i -= 1
                
                # Select control and targets
                list = [i, i+1]
                random_control = random.choice(list)
                random_target = list[1] if random_control == list[0] else list[0]

                self.circuit_matrix[random_control, d]=("", random_control, [], [], [], []) # We empty this cell for the CNOT 
                self.circuit_matrix[random_target, d]=("CNOT", random_target, [random_control,random_target], [], [random_control], [random_target])

            # Rebuild the circuit and evaluate fitness    
            self.tensor_to_circuit()
            self.fitness()


    def calculate_ground_state_energy(self):
        # Transform the qml.Hamiltonian objects to np matrices
        wires = [i for i in range(self.num_qubits)]
        H_matrix = qml.matrix(self.hamiltonian, wire_order=wires)

        # Diagonalize the Hamiltonian to find the ground state
        eigenvalues, _ = np.linalg.eigh(H_matrix) # This returns the sorted eigenvalues of the matrix
        ground_state_energy = eigenvalues[0] # The first eigenvalue is the ground state energy
        
        return ground_state_energy

    
class Evolutionary_Algorithm():
    
    '''
    This class handles the execution of the evolutionary algorithm.
    '''
    
    def __init__(self, num_circuits, num_of_generations, mutation_rate, num_qubits, depth, hamiltonian_label, h = None, max_mutation_rate=0.9, min_mutation_rate=0.1):
        self.num_circuits = num_circuits
        self.num_of_generations = num_of_generations
        self.num_qubits = num_qubits
        self.depth = depth
        self.hamiltonian_label = hamiltonian_label        
        self.best_circuit = None
        self.circuits = []
        self.best_energy = np.inf # initialise infinite energy, since the GA tries to minimize the energy
        if h is not None:
            self.h = h # Transverse field coefficient
        self.energies = []

        # Fitness-stagnation adaptation parameters --- Could convert most of these to hyperparameters if we want
        self.stagnation_threshold = 3  # Generations without improvement
        self.stagnation_counter = 0
        self.prev_best_energy = float('inf')
        self.mutation_rate = mutation_rate  # Initial rate
        self.max_mutation_rate = max_mutation_rate 
        self.min_mutation_rate = min_mutation_rate

    def selection(self, tournament_size=5, num_selected=None):
        """
        Tournament selection: Randomly pick 'tournament_size' individuals, 
        select the one with the lowest energy (best fitness).
        Repeat until 'num_selected' individuals are chosen.
        """
        if num_selected is None:
            num_selected = len(self.circuits)
        selected = []
        population = self.circuits.copy()  # Create a copy of the population to avoid modifying the original list

        for _ in range(num_selected):
            # Randomly sample without replacement for each tournament
            tournament = random.sample(population, tournament_size)
            # Select the individual with the lowest energy (best fitness)
            winner = min(tournament, key=lambda c: c.energy)
            population.remove(winner)  # Remove the winner from the population to avoid reselection
            selected.append(winner)
        return selected
    
    def sigmoid(self, x):
        """
        Sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def perform_evolution(self, print_circuit=True, vqe=False):
        '''
        This method performs the evolutionary algorithm by initialising a first generation
        of circuits, and then defining the 'Selection > Crossover > Mutation Logic' for all 
        subsequent generations.
        '''
                
        #### Initial Population ####
        for i in range(self.num_circuits):
            self.circuits.append(VQE_ansatz(num_qubits = self.num_qubits, depth = self.depth, hamiltonian_label = self.hamiltonian_label, h = self.h))
            if self.circuits[i].energy < self.best_energy:
                self.best_energy = self.circuits[i].energy # defines the initial best energy
                self.best_circuit = self.circuits[i]

        #### Subsequent Populations ####
        
        for gen in range(self.num_of_generations):
            
            self.circuits.sort(key = lambda x: x.energy,reverse = False) # sort existing circuits from smallest to largest energy           
            
            if len(self.circuits) % 2 != 0:
                self.circuits = self.circuits[:-1]

            # Tournament selection with elitism
            parents = self.selection(tournament_size=5, num_selected=int(0.4*self.num_circuits))
            probability_entanglement_aware = 2*(self.sigmoid(gen/2)-0.5) # the probability for the crossover follows a sigmoid function in the number of generations 
            new_population = parents.copy()
            
            while len(new_population) < self.num_circuits:
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = parent1.crossover(parent2, probability_entanglement_aware)
                child1.mutation(self.mutation_rate)
                child2.mutation(self.mutation_rate)
                new_population.extend([child1, child2])
            
            assert len(new_population) == self.num_circuits or len(new_population) == self.num_circuits+1, "New population size does not match the expected number of circuits."
            self.circuits = new_population[:self.num_circuits] 
            
            current_best_energy = min(c.energy for c in self.circuits)
            current_best_circuit = min(self.circuits, key=lambda c: c.energy)
            self.prev_best_energy=self.best_energy #saving prev_best_energy
            if current_best_energy < self.best_energy:
                self.best_energy = current_best_energy
                self.best_circuit = current_best_circuit
            
            if print_circuit:
                print(f"\nGeneration {gen} has best energy {self.best_energy:.4f}, best circuit: [mutation rate={self.mutation_rate:.4f}]")
                self.best_circuit.visualize_circuit()
                print()

            self.energies.append(self.best_energy)

            # Track stagnation
            energy_scale = abs(self.best_energy) if self.best_energy != 0 else 1
            stagnation_threshold = max(1e-5, energy_scale * 0.01)  # 1% of current energy
            if abs(self.best_energy - self.prev_best_energy) < stagnation_threshold:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                self.prev_best_energy = self.best_energy

            # Adapt mutation rate
            if self.stagnation_counter > self.stagnation_threshold:
                self.mutation_rate = min(self.mutation_rate * 2.0, self.max_mutation_rate)
            else:
                self.mutation_rate = max(self.mutation_rate * 0.95, self.min_mutation_rate)
            print(f"Best energy after generation {gen}: {self.best_circuit.energy}, mutation rate {self.mutation_rate}")
        print(f"Best circuit after evolution: ")        
        self.best_circuit.visualize_circuit()
        
        #### VQE ####

        if vqe == True:
            param_indices = []
            init_params = []
            for d in range(self.best_circuit.circuit_matrix.shape[1]):
                for q in range(self.best_circuit.circuit_matrix.shape[0]):
                    gate = self.best_circuit.circuit_matrix[q, d]
                    if gate["name"] in ["RX", "RY", "RZ"]:
                        param_indices.append((q, d))
                        init_params.append(gate["parameters"][0])
            init_params = np.array(init_params, dtype=float)
            init_params = qml.numpy.array(init_params, requires_grad=True)

            if len(init_params) == 0:
                return self.energies, self.best_energy, self.best_circuit

            dev = qml.device("default.qubit", wires=self.num_qubits)

            def circuit(params):
                idx = 0
                for d in range(self.best_circuit.circuit_matrix.shape[1]):
                    for q in range(self.best_circuit.circuit_matrix.shape[0]):
                        gate = self.best_circuit.circuit_matrix[q, d]
                        if gate["name"] == "RX":
                            qml.RX(params[idx], wires=q)
                            idx += 1
                        elif gate["name"] == "RY":
                            qml.RY(params[idx], wires=q)
                            idx += 1
                        elif gate["name"] == "RZ":
                            qml.RZ(params[idx], wires=q)
                            idx += 1
                        elif gate["name"] == "CNOT":
                            qml.CNOT(wires=gate["control_qubits"] + gate["target_qubits"])
                        elif gate["name"] == "Id":
                            qml.Identity(wires=q)
                        elif gate["name"] == "H":
                            qml.Hadamard(wires=q)
                return qml.expval(self.best_circuit.hamiltonian)

            qnode = qml.QNode(circuit, dev)

            opt = qml.AdamOptimizer(stepsize=0.05)
            params = init_params.copy()
            max_iterations = 151
            assert abs(qnode(init_params) - self.best_energy) < 1e-10 
            print(f'\nBest circuit energy before VQE optimization: {self.best_energy:.4f}')
            
            for i in range(max_iterations):
                if i % 10 == 0 and i > 0:
                    print(f"VQE Energy at Step {i} {energy:.6f}")
                params, energy = opt.step_and_cost(qnode, params)
                
            for idx, (q, d) in enumerate(param_indices):
                gate = list(self.best_circuit.circuit_matrix[q, d])
                gate[3] = [params[idx]]  # Update parameters (4th element in each tuple)
                self.best_circuit.circuit_matrix[q, d] = tuple(gate)
            self.best_circuit.tensor_to_circuit()
            self.best_circuit.fitness()
            self.best_energy = self.best_circuit.energy
                
        return self.energies, self.best_energy, self.best_circuit
 
        