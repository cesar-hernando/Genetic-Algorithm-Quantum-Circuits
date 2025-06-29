'''
In thi script, we will implement a hyperparameter optimization process over the depth, mutation rate, number of circuits and number of generations for the VQE ansatz. The goal is to find the best combination of these parameters that minimizes the energy of the quantum circuit.
'''

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from gen_alg_VQE import Evolutionary_Algorithm, VQE_ansatz
import os

##########################################################################################################
###################### Hyperparameter Optimization (Tuning one parameter at a time) ######################
##########################################################################################################

def mutation_rate_optimization(mutation_rates, num_qubits, depth, num_circuits, num_generations, hamiltonian_label, h):
    """
    By sweeping through a range of mutation rates, the optimal value is given by the one that yields the lowest energy.
    """
    best_energies = []

    for mutation_rate in tqdm(mutation_rates, desc="Testing mutation rates"):
        evol_alg = Evolutionary_Algorithm(num_circuits, num_generations, mutation_rate, num_qubits, depth, hamiltonian_label, h)
        _, best_energy, _ = evol_alg.perform_evolution(print_circuit=False)
        best_energies.append(best_energy)

    optimal_mutation_rate = mutation_rates[np.argmin(best_energies)]

    return  best_energies, optimal_mutation_rate


def depth_optimization(depths, num_qubits, mutation_rate, num_circuits, num_generations, hamiltonian_label, h):
    """
    By sweeping through a range of depths, the optimal value is given by the one that yields the lowest energy.
    """
    best_energies = []

    for depth in tqdm(depths, desc="Testing depths"):
        evol_alg = Evolutionary_Algorithm(num_circuits, num_generations, mutation_rate, num_qubits, depth, hamiltonian_label, h)
        _, best_energy, _ = evol_alg.perform_evolution(print_circuit=False)
        best_energies.append(best_energy)

    optimal_depth = depths[np.argmin(best_energies)]

    return  best_energies, optimal_depth


def num_circuits_optimization(num_circuits_list, num_qubits, mutation_rate, depth, num_generations, hamiltonian_label, h):
    """
    By sweeping through a range of number of circuits, the optimal value is given by the one that yields the lowest energy.
    """
    best_energies = []

    for num_circuits in tqdm(num_circuits_list, desc="Testing number of circuits"):
        evol_alg = Evolutionary_Algorithm(num_circuits, num_generations, mutation_rate, num_qubits, depth, hamiltonian_label, h)
        _, best_energy, _ = evol_alg.perform_evolution(print_circuit=False)
        best_energies.append(best_energy)

    optimal_num_circuits = num_circuits_list[np.argmin(best_energies)]

    return  best_energies, optimal_num_circuits


def num_generations_optimization(num_generations_list, num_qubits, mutation_rate, depth, num_circuits, hamiltonian_label, h):
    """
    By sweeping through a range of number of generations, the optimal value is given by the one that yields the lowest energy.
    """
    best_energies = []

    for num_generations in tqdm(num_generations_list, desc="Testing number of generations"):
        evol_alg = Evolutionary_Algorithm(num_circuits, num_generations, mutation_rate, num_qubits, depth, hamiltonian_label, h)
        _, best_energy, _ = evol_alg.perform_evolution(print_circuit=False)
        best_energies.append(best_energy)

    optimal_num_generations = num_generations_list[np.argmin(best_energies)]

    return  best_energies, optimal_num_generations


def plot_energies_hyperparameter(best_energies, hyperparameter_list, xlabel, num_qubits, depth, hamiltonian_label):
    plt.figure(figsize=(10, 6))
    plt.plot(hyperparameter_list, best_energies, marker='.')
    plt.title(f'Estimated GS Energy vs {xlabel} for {num_qubits} Qubits and Depth {depth} ({hamiltonian_label})')
    plt.xlabel(xlabel)
    plt.ylabel('Energy')
    plt.savefig(f"plots_hyperparameter_optimization/hyperparam_dependency_{xlabel}.png")
    plt.close()

# Define the default values of the parameters when they are not varied in the optimization process
num_qubits = 4
depth = 6
num_circuits = 75
num_generations = 50
mutation_rate = 0.5
hamiltonian_label = 'transverse_field_ising'
h = 0 # choose from np.linspace(0,3,10)

# Modify the following for the optimisation of the different hyperparameters
mutation_rate_analysis = False
depth_analysis = False
num_circuits_analysis = False
num_generations_analysis = False

if mutation_rate_analysis:
    # Mutation Rate Optimization
    print(">>> Running mutation rate optimization ... \n")
    mutation_rates = np.linspace(0.1, 0.9, 5)
    best_energies_mutation, optimal_mutation_rate = mutation_rate_optimization(mutation_rates, num_qubits, depth, num_circuits, num_generations, hamiltonian_label, h)
    plot_energies_hyperparameter(best_energies_mutation, mutation_rates, 'Mutation Rate', num_qubits, depth, hamiltonian_label) 
    print(f"Optimal Mutation Rate: {optimal_mutation_rate}\n")

if depth_analysis:
    # Depth Optimization
    print(">>> Running depth optimization ... \n")
    depths = np.arange(3, 9)
    best_energies_depth, optimal_depth = depth_optimization(depths, num_qubits, mutation_rate, num_circuits, num_generations, hamiltonian_label, h)
    plot_energies_hyperparameter(best_energies_depth, depths, 'Depth', num_qubits, depth, hamiltonian_label)
    print(f"Optimal Depth: {optimal_depth}\n")

if num_circuits_analysis:
    # Number of Circuits Optimization
    print(">>> Running number of circuits optimization ... \n")
    num_circuits_list = np.arange(50, 111, 10)
    best_energies_circuits, optimal_num_circuits = num_circuits_optimization(num_circuits_list, num_qubits, mutation_rate, depth, num_generations, hamiltonian_label, h)
    plot_energies_hyperparameter(best_energies_circuits, num_circuits_list, 'Number of Circuits', num_qubits, depth, hamiltonian_label)
    print(f"Optimal Number of Circuits: {optimal_num_circuits}\n")

if num_generations_analysis:
    # Number of Generations Optimization
    print(">>> Running number of generations optimization ... \n")
    num_generations_list = np.arange(30, 91, 10)  
    best_energies_generations, optimal_num_generations = num_generations_optimization(num_generations_list, num_qubits, mutation_rate, depth, num_circuits, hamiltonian_label, h)
    plot_energies_hyperparameter(best_energies_generations, num_generations_list, 'Number of Generations', num_qubits, depth, hamiltonian_label)
    print(f"Optimal Number of Generations: {optimal_num_generations}\n")

ground_state_energy = VQE_ansatz(num_qubits, depth, hamiltonian_label, h).calculate_ground_state_energy()
print(f"Ground State Energy = {ground_state_energy}\n")


##########################################################################################################
######################### Two hyperparameter optimisation ################################################
##########################################################################################################


def two_hyperparam_optimisation(varied_parameters, varied_parameters_labels, fixed_params, fixed_params_labels, num_qubits, hamiltonian_label, h):
    """
    varied_parameters: np.array
        2D numpy array where each row corresponds to a different hyperparameter to be varied.

    varied_parameters_labels: string list
        List of labels for the varied parameters.
        
    fixed_params: np.array
        List of parameter that is fixed
    
    fixed_params_labels: string list
        List of label for the fixed parameters.
    
    num_qubits: int
        Number of qubits used in VQE ansatz.
    
    hamiltonian_label: str
        Label of the measurement Hamiltonian.
    
    h: float
        Transversal strength of magnetic field in Ising model.
    """

    all_best_energies = []
    all_parameters={"num_circuits": None, "num_generations": None, "depth": None}
    print(all_parameters)
    print(fixed_parameters)
    print(fixed_parameters_labels)
    all_parameters[fixed_params_labels[0]] = fixed_params[0]
    
    for param1 in tqdm(varied_parameters[0, :], desc=f"Testing {varied_parameters_labels[0]}"):
        local_best_energies=[]
        for param2 in tqdm(varied_parameters[1, :], desc=f"Testing {varied_parameters_labels[1]}"):
            all_parameters[varied_parameters_labels[0]] = param1
            all_parameters[varied_parameters_labels[1]] = param2
            print(f">>> Running with {varied_parameters_labels[0]} = {param1}, {varied_parameters_labels[1]} = {param2} ...")
            evol_alg = Evolutionary_Algorithm(num_circuits=int(all_parameters["num_circuits"]), num_of_generations=int(all_parameters["num_generations"]), mutation_rate = mutation_rate, num_qubits=num_qubits, depth=int(all_parameters["depth"]), hamiltonian_label=hamiltonian_label, h=h)
            _, best_energy, _ = evol_alg.perform_evolution(print_circuit=False)
            print(f"Best energy found: {best_energy:.4f}\n")
            local_best_energies.append(best_energy)
        all_best_energies.append(local_best_energies)
    all_best_energies = np.array(all_best_energies)

    # Determining optimal parameters
    min_idx = np.unravel_index(np.argmin(all_best_energies), all_best_energies.shape)
    optimal_param1 = varied_parameters[0, min_idx[0]]
    optimal_param2 = varied_parameters[1, min_idx[1]]
    optimal_energy = all_best_energies[min_idx]
    optimal_params = all_parameters.copy()
    optimal_params[varied_parameters_labels[0]] = optimal_param1
    optimal_params[varied_parameters_labels[1]] = optimal_param2
    
    return all_best_energies, optimal_params, min_idx, (optimal_param1, optimal_param2), optimal_energy

def two_hyperparam_heatmap(best_energies, varied_parameters, varied_parameters_labels, varied_opt_idx, optimal_params, optimal_energy, h):
    """
    Plots a heatmap of the best energies for the two varied parameters.
    """
    plt.figure(figsize=(10, 6))
    best_energies = np.array(best_energies, dtype=float) 
    vmin = np.min(best_energies)
    vmax = np.max(best_energies)
    im = plt.imshow(best_energies, aspect='auto', cmap='viridis', origin='lower', vmin=np.min(best_energies), vmax=np.max(best_energies))
    plt.scatter(varied_opt_idx[1], varied_opt_idx[0], color='red', label=f'Optimal Energy: {optimal_energy:.4f} for {varied_parameters_labels[0]} = {optimal_params[0]:.1f}, {varied_parameters_labels[1]} = {optimal_params[1]:.1f}', marker='x', s=100)
    plt.legend(loc='upper right')
    cbar = plt.colorbar(im, label='Best Energy')
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
    plt.xticks(ticks=np.arange(len(varied_parameters[1])), labels=np.char.mod('%.1f', varied_parameters[1]), rotation=45)
    plt.yticks(ticks=np.arange(len(varied_parameters[0])), labels=np.char.mod('%.1f', varied_parameters[0]))
    plt.xlabel(varied_parameters_labels[1])
    plt.ylabel(varied_parameters_labels[0])
    plt.title(f'Best Energies for varying {varied_parameters_labels[0]} and {varied_parameters_labels[1]}')
    plt.tight_layout()
        
    directory = f"plots_hyperparameter_optimization/h{h}"
    filename = f"two_hyperparam_heatmap_{varied_parameters_labels[0]}_{varied_parameters_labels[1]}.png"
    filepath = os.path.join(directory, filename)

    if not os.path.exists(directory):
        os.makedirs(directory)
        
    plt.savefig(filepath)
    plt.close()

transverse_field_strengths1 = np.linspace(0,3,20)[:4] # Cesar
transverse_field_strengths2 = np.linspace(0,3,20)[4:8] # Arturo
transverse_field_strengths3 = np.linspace(0,3,20)[8:12] # Lukas
transverse_field_strengths4 = np.linspace(0,3,20)[12:16] # Badr
transverse_field_strengths5 = np.linspace(0,3,20)[16:] # Tim

for h in transverse_field_strengths1:
    mutation_rate = 0.7 # fixed parameter 
    # Default parameters for the two hyperparameter optimization
    num_qubits = 4
    hamiltonian_label = 'transverse_field_ising'
    depth = 6
    num_circuits = 75
    num_generations = 50
    default_parameters = {"num_circuits": num_circuits, "num_generations": num_generations, "depth": depth}

    num_points_per_parameter = 8 # The following parameters have the same length for an easier implementation
    depths_analysis= np.linspace(3, 10, num_points_per_parameter) 
    num_circuits_analysis= np.linspace(30, 100, num_points_per_parameter) 
    num_generations_analysis=np.linspace(30, 100, num_points_per_parameter)

    print(depths_analysis, num_circuits_analysis, num_generations_analysis)

    error_text = "All parameter arrays must have the same length for two hyperparameter optimization."
    assert len(depths_analysis) == len(num_circuits_analysis) == len(num_generations_analysis) == num_points_per_parameter, error_text

    parameters= {"depth": depths_analysis, "num_circuits": num_circuits_analysis, "num_generations": num_generations_analysis}
    parameter_labels = ["num_circuits", "num_generations", "depth"]

    global_best_energies = []
    global_optimal_params = []
    global_best_energy = np.inf
    num_precision = 0.001 # Precision for the energy values, used to compare energies
    
    # Iterate through all pairs of parameters to perform two hyperparameter optimization
    for i, label1 in enumerate(parameter_labels):
        for j, label2 in enumerate(parameter_labels[i+1:]):
            varied_parameters = np.array([parameters[label1], parameters[label2]])
            varied_parameters_labels = [label1, label2]
            fixed_parameters_labels = [label for label in parameter_labels if label not in varied_parameters_labels]
            # Create a fixed parameters array with the default values for the fixed parameters
            fixed_parameters = np.array([default_parameters[label] for label in fixed_parameters_labels])
            print(f">>> Running two hyperparameter optimization for {varied_parameters_labels[0]} and {varied_parameters_labels[1]}, Fixed parameters: {fixed_parameters_labels} = {fixed_parameters} ... \n")
            best_energies, optimal_params, varied_opt_idx, varied_optimal_params, optimal_energy = two_hyperparam_optimisation(varied_parameters, varied_parameters_labels, fixed_parameters, fixed_parameters_labels, num_qubits, hamiltonian_label, h)

            # Obtain the optimal parameters and energy
            if optimal_energy < global_best_energy - num_precision: # Update global best energy and parameters if a better one is found
                global_best_energy = optimal_energy
                global_best_energies = [optimal_energy] # Remove previous global best energies
                global_optimal_params = [optimal_params] # Remove previous global optimal parameters
            elif abs(optimal_energy - global_best_energy) < num_precision: # If the energy is within the precision range, append the parameters
                global_optimal_params.append(optimal_params) # Append to the list of global optimal parameters
                global_best_energies.append(optimal_energy) # Append to the list of global best energies

            # Plot the heatmap for the current pair of parameters
            two_hyperparam_heatmap(best_energies, varied_parameters, varied_parameters_labels, varied_opt_idx, varied_optimal_params, optimal_energy, h)
                

    print(f"\n>>> Global best energy found: {global_best_energy:.4f}")
    print(f"\n>>> All optimal parameters with their associated energies:\n")
    for i, params in enumerate(global_optimal_params):
        print(f"{params} with energy {global_best_energies[i]:.4f}")
    print("\n>>> Hyperparameter optimization completed.\n")