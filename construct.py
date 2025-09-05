def construct_hmm(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    T = int(lines[0].strip())
    
    transition_counts = {}
    emission_counts = {}
    all_states = set()
    all_observables = set()
    
    line_idx = 1
    for _ in range(T):
        # states
        states = list(map(int, lines[line_idx].strip().split()))
        # observables
        observables = list(map(int, lines[line_idx + 1].strip().split()))
        line_idx += 2
        
        all_states.update(states)
        all_observables.update(observables)
        
        #(state[i] to state[i+1])
        for i in range(len(states) - 1):
            from_state = states[i]
            to_state = states[i + 1]
            if from_state not in transition_counts:
                transition_counts[from_state] = {}
            if to_state not in transition_counts[from_state]:
                transition_counts[from_state][to_state] = 0
            transition_counts[from_state][to_state] += 1
        
        #(state to observable)
        for state, observable in zip(states, observables):
            if state not in emission_counts:
                emission_counts[state] = {}
            if observable not in emission_counts[state]:
                emission_counts[state][observable] = 0
            emission_counts[state][observable] += 1
    
    states_list = sorted(list(all_states))
    observables_list = sorted(list(all_observables))
    
    N = len(states_list)  # total states
    M = len(observables_list)  # total observables
    
    state_to_idx = {state: i for i, state in enumerate(states_list)}
    obs_to_idx = {obs: i for i, obs in enumerate(observables_list)}
    
    A = [[0.0 for _ in range(N)] for _ in range(N)]
    B = [[0.0 for _ in range(M)] for _ in range(N)]
    
    #matrix A
    for from_state in transition_counts:
        from_idx = state_to_idx[from_state]
        total_transitions = sum(transition_counts[from_state].values())
        
        for to_state in transition_counts[from_state]:
            to_idx = state_to_idx[to_state]
            A[from_idx][to_idx] = transition_counts[from_state][to_state] / total_transitions
    
    #matrix B
    for state in emission_counts:
        state_idx = state_to_idx[state]
        total_emissions = sum(emission_counts[state].values())
        
        for obs in emission_counts[state]:
            obs_idx = obs_to_idx[obs]
            B[state_idx][obs_idx] = emission_counts[state][obs] / total_emissions
    
    return A, B, states_list, observables_list

def save_matrices(A, B, filename):
    print(f"Saving matrices to {filename}...")
    
    with open(filename, 'w') as f:
        # A
        for row in A:
            formatted_row = []
            for val in row:
                if val == 0:
                    formatted_row.append('0')
                else:
                    formatted_val = f"{val:.10f}".rstrip('0').rstrip('.')
                    formatted_row.append(formatted_val)
            f.write(' '.join(formatted_row) + '\n')
        
        # B  
        for row in B:
            formatted_row = []
            for val in row:
                if val == 0:
                    formatted_row.append('0')
                else:
                    formatted_val = f"{val:.10f}".rstrip('0').rstrip('.')
                    formatted_row.append(formatted_val)
            f.write(' '.join(formatted_row) + '\n')
    
    print(f"Matrices saved successfully!")

if __name__ == "__main__":
    import sys
    
    try:
        print("="*50)
        print("PROCESSING DATASET 1")
        print("="*50)
        A1, B1, states1, obs1 = construct_hmm("dataset1.in")
        save_matrices(A1, B1, "sol-1-1.txt")
        print(f"Dataset 1 - States: {states1}")
        print(f"Dataset 1 - Observables: {obs1}")
        print(f"Transition matrix A1 shape: {len(A1)}x{len(A1[0])}")
        print(f"Emission matrix B1 shape: {len(B1)}x{len(B1[0])}")
        
        print("\n" + "="*50)
        print("PROCESSING DATASET 2")
        print("="*50)
        
        A2, B2, states2, obs2 = construct_hmm("dataset2.in")
        save_matrices(A2, B2, "sol-1-2.txt")
        print(f"Dataset 2 - States: {states2}")
        print(f"Dataset 2 - Observables: {obs2}")
        print(f"Transition matrix A2 shape: {len(A2)}x{len(A2[0])}")
        print(f"Emission matrix B2 shape: {len(B2)}x{len(B2[0])}")
        
        print("\n" + "="*50)
        print("HMM CONSTRUCTION COMPLETED FOR BOTH DATASETS!")
        print("="*50)
        
    except FileNotFoundError as e:
        print(f"Error: Dataset file not found - {e}")
        print("Please ensure dataset1.txt and dataset2.txt are in the same directory")
    except Exception as e:
        print(f"Error during HMM construction: {e}")
        import traceback
        traceback.print_exc()