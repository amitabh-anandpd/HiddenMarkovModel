import math
def load_matrices(filename):
    with open("sol-1-1.txt", 'r') as f:
        raw_lines = f.readlines()
    lines = [line for line in raw_lines if line.strip()]
        
    total_lines = len(lines)
    N = total_lines//2
    
    # Read matrix A (first N lines)
    A = []
    for i in range(N):
        row = list(map(float, lines[i].split()))
        A.append(row)
    
    # Read matrix B (remaining lines)
    B = []
    for i in range(N, total_lines):
        row = list(map(float, lines[i].split()))
        B.append(row)
    
    return A, B

def viterbi(observations, A, B, start_probs=None):
    """
    Viterbi algorithm to find most likely sequence of hidden states
    
    Args:
        observations: sequence of observed symbols (0-indexed)
        A: transition probability matrix (N x N)
        B: emission probability matrix (N x M)  
        start_probs: initial state probabilities (if None, uniform distribution)
    
    Returns:
        Most likely sequence of states
    """
    N = len(A)
    T = len(observations)
    
    if T == 0:
        return []
    
    if start_probs is None:
        # For HMM problems, typically we start from state 0 (Start state)
        # But we need to transition from Start to actual states
        start_probs = [0.0] * N
        start_probs[0] = 1.0  # Start at state 0
    
    # Initialize Viterbi tables
    viterbi_prob = [[0.0 for _ in range(T)] for _ in range(N)]
    viterbi_path = [[0 for _ in range(T)] for _ in range(N)]
    
    # Initialization step (t=0)
    for state in range(N):
        if observations[0] < len(B[state]):
            viterbi_prob[state][0] = start_probs[state] * B[state][observations[0]]
        else:
            viterbi_prob[state][0] = 0.0
        viterbi_path[state][0] = 0
    
    # Recursion step (t=1 to T-1)
    for t in range(1, T):
        for state in range(N):
            max_prob = 0.0
            max_prev_state = 0
            
            for prev_state in range(N):
                prob = viterbi_prob[prev_state][t-1] * A[prev_state][state]
                if prob > max_prob:
                    max_prob = prob
                    max_prev_state = prev_state
            
            if observations[t] < len(B[state]):
                viterbi_prob[state][t] = max_prob * B[state][observations[t]]
            else:
                viterbi_prob[state][t] = 0.0
            viterbi_path[state][t] = max_prev_state
    
    # Termination step - find the most likely final state
    max_final_prob = 0.0
    best_final_state = 0
    
    for state in range(N):
        if viterbi_prob[state][T-1] > max_final_prob:
            max_final_prob = viterbi_prob[state][T-1]
            best_final_state = state
    
    # Backtrack to find the most likely path
    best_path = [0] * T
    best_path[T-1] = best_final_state
    
    for t in range(T-2, -1, -1):
        best_path[t] = viterbi_path[best_path[t+1]][t+1]
    
    return best_path

def solve_predictions():
    """Main function to solve prediction problems"""
    print("Starting prediction process...")
    
    try:
        with open("input.txt", 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("Error: input.txt not found!")
        return
    
    lines = [line.strip() for line in lines if line.strip()]
    dataset_num = int(lines[0])
    num_test_cases = int(lines[1])
    
    print(f"Using dataset {dataset_num}")
    print(f"Number of test cases: {num_test_cases}")
    
    # Load appropriate HMM matrices
    if dataset_num == 1:
        A, B = load_matrices("sol-1-1.txt")
    elif dataset_num == 2:
        A, B = load_matrices("sol-1-2.txt")
    else:
        print(f"Error: Invalid dataset number {dataset_num}")
        return
    
    results = []
    line_idx = 2
    
    for test_case in range(num_test_cases):
        print(f"\nProcessing test case {test_case + 1}/{num_test_cases}")
        
        if line_idx >= len(lines):
            print(f"Error: Not enough lines in input file for test case {test_case + 1}")
            break
            
        num_observations = int(lines[line_idx])
        line_idx += 1
        
        if line_idx >= len(lines):
            print(f"Error: Missing observations for test case {test_case + 1}")
            break
            
        observations = list(map(int, lines[line_idx].split()))
        line_idx += 1
        
        if len(observations) != num_observations:
            print(f"Warning: Expected {num_observations} observations, got {len(observations)}")
        
        print(f"Observations: {observations}")
        
        # Validate observations are within range
        max_obs = len(B[0]) if B and B[0] else 0
        if any(obs >= max_obs for obs in observations):
            print(f"Warning: Some observations are out of range (max allowed: {max_obs - 1})")
            # Filter out invalid observations
            observations = [obs for obs in observations if obs < max_obs]
        
        if not observations:
            print("No valid observations for this test case")
            results.append("")
            continue
        
        try:
            # Find most likely state sequence using Viterbi
            best_states = viterbi(observations, A, B, A[0])
            print(f"Best state sequence: {best_states}")
            
            # Format result according to problem requirements
            # For HMM prediction problems, we usually want the actual hidden states
            # not including the start state (state 0) in the output
            if best_states:
                # Remove start state (0) if it appears at the beginning
                if len(best_states) > 0 and best_states[0] == 0:
                    result_states = best_states[1:] if len(best_states) > 1 else []
                else:
                    result_states = best_states
                
                if result_states:
                    results.append(' '.join(map(str, result_states)))
                else:
                    results.append("0")  # Default if no valid states
            else:
                results.append("0")
                
        except Exception as e:
            print(f"Error processing test case {test_case + 1}: {e}")
            results.append("0")  # Default fallback
    
    # Write results to output file
    try:
        with open("sol-2.txt", 'w') as f:
            for result in results:
                f.write(result + '\n')
        print(f"\nResults written to sol-2.txt")
        print("Results:")
        for i, result in enumerate(results):
            print(f"Test case {i+1}: {result}")
    except Exception as e:
        print(f"Error writing results: {e}")

if __name__ == "__main__":
    solve_predictions()
    print("Predictions completed!")