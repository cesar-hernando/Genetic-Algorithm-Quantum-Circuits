'''
This file drives the execution of the Evolutionary algorithm.
'''
#from ray import tune
from gen_alg_VQE import VQE_ansatz, Evolutionary_Algorithm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import pennylane as qml

#######################################
############ PARAMETERS ###############
#######################################

# uncommented values and arrays may be used for optimized 4-qubit run
num_qubits = 8
mutation_rate = 0.5
hamiltonian_label = 'transverse_field_ising'
transverse_field_strengths = np.linspace(0,2,10) # np.linspace(0,3,20)

# variable parameters (for each transverse field strength h) 
depths = [10]*10               #[4,7,7,5,9,9,9,8,8,9,8,7,7,6,6,10,9,9,9,9] 
num_circuits = [100]*10        #[80,70,40,60,90,80,90,100,80,90,60,40,80,100,90,90,90,90,100,100] 
num_of_generations = [250]*10  #[100,100,100,100,90,70,80,90,90,90,90,60,90,100,90,100,90,100,100,100] 

########### TESTING ############
testing = False
if testing:
    h = 0
    evol_alg = Evolutionary_Algorithm(num_circuits[0], num_of_generations[0], mutation_rate, num_qubits, depths[0], hamiltonian_label, h)
    energies, best_energy, best_circuit = evol_alg.perform_evolution(print_circuit=True)
    gs_energy = VQE_ansatz(num_qubits, depths[0], 'transverse_field_ising', h).calculate_ground_state_energy()
    print(f'\n > Ground state energy for transverse field Ising Hamiltonian with transverse field strength h = {h:.2f} on {num_qubits} qubits: {gs_energy:.2f}')
    print(f"\nBest EA energy for transverse field Ising Hamiltonian with transverse field strength h = {h:.2f} =  {best_energy:.2f}, achieved by circuit: \n")
    best_circuit.visualize_circuit()
    quit() # Stop execution of the script

#######################################
############ PLOTTING #################
#######################################

def plot_energies_each_h(energies, gs_energy, gs_difference, energies_std, gs_difference_std, h, num_of_generations):
    """
    Plots the best energy of each generation for a given transverse field strength h.
    """

    fig, axs = plt.subplots(2, 1, figsize=(10, 15), sharex=True)
    fig.suptitle(f'Transverse-field Ising Hamiltonian with h = {h:.2f} on {num_qubits} qubits', size = 24)
    # absolute energy subplot
    axs[0].errorbar(range(num_of_generations), energies, yerr=energies_std, marker='o', linestyle="-", color="b", capsize=5)
    axs[0].axhline(gs_energy, color='green', linestyle='--', label=r'$E_{GS}$ for transverse-field Ising Hamiltonian')
    axs[0].legend(fontsize=16)  
    axs[0].set_ylabel('Energy', size = 18)
    axs[0].set_title(f'Evolution of minimal energy (EA)', size = 20)
   
    axs[0].tick_params(axis='both', which='major', labelsize=14)
    axs[0].legend(loc='upper right')

    # Energy difference subplot
    axs[1].errorbar(range(num_of_generations), gs_difference, yerr=gs_difference_std, marker='o', linestyle="-", color="r", capsize=5)
    axs[1].set_xlabel('Generation', size = 18)
    axs[1].set_ylabel(r'$E_{EA}-E_{GS}$', size = 18)
    axs[1].tick_params(axis='both', which='major', labelsize=14)
    axs[1].set_title(f'Evolution of energy difference', size = 20)
    axs[1].set_yscale('log')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.savefig(f"plots_optimality/transverse_ising_{h:.2f}.png")
    plt.close()
    
def plot_energies_all_h(transverse_field_strengths, best_energies, best_energies_std, energy_differences, energy_differences_std, gs_energies):

    ### Top: Ground state energy and EA energy per transverse field strength. Bottom: Energy difference over transverse field strengths.
    # Plotting of h-dependence

    fig, axs = plt.subplots(2, 1, figsize=(7, 10), sharex=True)
    fig.suptitle(f'Transverse-field Ising Hamiltonian on {num_qubits} qubits', size = 18)

    # Absolute energy subplot
    axs[0].errorbar(transverse_field_strengths, best_energies, yerr=best_energies_std, label = r'Optimal energies $E_{EA}$')
    axs[0].plot(transverse_field_strengths, gs_energies, label = r'Ground state energies $E_{GS}$')
    axs[0].set_ylabel('Energy', size = 18)
    axs[0].tick_params(axis='both', which='major', labelsize=14)
    axs[0].legend(loc='upper right')

    # Energy difference subplot

    axs[1].errorbar(transverse_field_strengths, energy_differences, yerr=energy_differences_std, )
    axs[1].set_xlabel('Transverse field strength h ', size = 18)
    axs[1].tick_params(axis='both', which='major', labelsize=14)
    axs[1].set_ylabel(r'$E_{EA}-E_{GS}$', size = 18)
    axs[1].set_yscale('log')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"plots_optimality/transverse_ising_h_sweep.png")
    plt.close()

def plot_z_magnetization(transverse_field_strengths, z_magnetizations, z_magnetizations_std):

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(transverse_field_strengths, np.abs(np.array(z_magnetizations)), yerr=z_magnetizations_std, marker='o', label='z-magnetization')
    plt.title(f'Z-magnetization of best circuits (transverse-field Ising Hamiltonian)', size = 20)
    plt.xlabel('Transverse field strength h', size = 18)
    plt.ylabel('Z-magnetization (absolute value)', size = 18)
    plt.xticks(size = 18)
    plt.yticks(size = 18) 
    plt.savefig(f"plots_optimality/z_magnetizations.png")
    plt.close()
    
    
#######################################
############ EXECUTION ################
#######################################

print("\n>>> Running Evolutionary Algorithm \n")

gs_energies = []
gs_diffs=[]
best_energies = []
best_energies_std=[]
energy_differences = []
energy_diffs_std=[]
best_circuits = []
z_magnetizations = []
z_magnetizations_std=[]
num_repeats = 1

for i, h in enumerate(transverse_field_strengths):
    
    same_h_all_energies = np.zeros(shape=(num_repeats, num_of_generations[i]))
    same_h_best_energies = []
    same_h_z_magnetization = []

    gs_transverse_ising = VQE_ansatz(num_qubits, depths[i], 'transverse_field_ising', h).calculate_ground_state_energy()
    gs_energies.append(gs_transverse_ising)
    print(f'>Ground state energies for transverse field Ising Hamiltonian with transverse field strength h = {h:.2f} on {num_qubits} qubits: {gs_transverse_ising:.2f}')


    for idx in range(num_repeats):
        print(f"\n>> Current transverse field strength {h} and iteration {idx+1} (out of {num_repeats})  \n")
        evol_alg = Evolutionary_Algorithm(num_circuits[i], num_of_generations[i], mutation_rate, num_qubits, depths[i], hamiltonian_label, h)
        energies, best_energy, best_circuit = evol_alg.perform_evolution(print_circuit=False, vqe=True) # can opt to perform VQE optimization on best circuit by setting vqe = True
        print(f"Best EA energy for transverse field Ising Hamiltonian with transverse field strength h = {h:.2f}: {best_energy:.2f}, achieved by circuit: \n")
        best_circuit.visualize_circuit()
        best_circuit.build_magnetization() 
        best_circuit.tensor_to_circuit(magnetization=True) 
        z_magnetization = best_circuit.qnode()
        
        print(f"\nz_magnetization of best circuit: {z_magnetization:.2f}\n")

        same_h_all_energies[idx, :] = energies
        same_h_best_energies.append(energies[-1])
        same_h_z_magnetization.append(z_magnetization)
        
    energies=np.mean(same_h_all_energies, axis=0)
    energies_std=np.std(same_h_all_energies, axis=0)
    gs_diffs.append(np.mean(same_h_all_energies-gs_energies[-1], axis=0))
    gs_diffs_std=np.std(same_h_all_energies - gs_energies[-1], axis=0)
    
    plot_energies_each_h(energies, gs_energies[-1], gs_diffs[-1], energies_std, gs_diffs_std, h, num_of_generations[i])
    
    best_energies.append(energies[-1])
    best_energies_std.append(energies_std[-1])

    energy_differences.append(energies[-1]-gs_transverse_ising)
    energy_diffs_std.append(gs_diffs_std[-1])

    z_magnetizations.append(np.mean(np.abs(np.array(same_h_z_magnetization))))
    z_magnetizations_std.append(np.std(np.abs(np.array(same_h_z_magnetization))))

plot_energies_all_h(transverse_field_strengths, best_energies, best_energies_std, energy_differences, energy_diffs_std, gs_energies)
plot_z_magnetization(transverse_field_strengths, z_magnetizations, z_magnetizations_std)

