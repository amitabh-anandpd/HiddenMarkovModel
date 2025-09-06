import math
def load_matrices(filename):
    with open("sol-1-1.txt", 'r') as f:
        raw_lines = f.readlines()
    lines = [line for line in raw_lines if line.strip()]
        
    total_lines = len(lines)
    N = total_lines//2
    
    #A (first N lines)
    A = []
    for i in range(N):
        row = list(map(float, lines[i].split()))
        A.append(row)
    
    #B (remaining lines)
    B = []
    for i in range(N, total_lines):
        row = list(map(float, lines[i].split()))
        B.append(row)
    
    return A, B

def viterbi(observations, A, B, start_probs=None):
    #implementation of viterbi algorithm (brute force is impossible)
    N = len(A)
    T = len(observations)
    
    if T == 0:
        return []
    
    if start_probs is None:
        print("Probability of going from state 0 to all other states not provided!!")
        return None
    
    #initialize tables
    viterbi_prob = [[0.0 for _ in range(T)] for _ in range(N)]
    viterbi_path = [[0 for _ in range(T)] for _ in range(N)]
    
    #initialization t=0
    for state in range(N):
        if observations[0] < len(B[state]):
            viterbi_prob[state][0] = start_probs[state] * B[state][observations[0]]
        else:
            viterbi_prob[state][0] = 0.0
        viterbi_path[state][0] = 0
    
    #recursion t=1 to T-1
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
    
    max_final_prob = 0.0
    best_final_state = 0
    
    for state in range(N):
        if viterbi_prob[state][T-1] > max_final_prob:
            max_final_prob = viterbi_prob[state][T-1]
            best_final_state = state
    
    #backtrack
    best_path = [0] * T
    best_path[T-1] = best_final_state
    
    for t in range(T-2, -1, -1):
        best_path[t] = viterbi_path[best_path[t+1]][t+1]
    
    return best_path

def solve_predictions():
    
    try:
        with open("input.txt", 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("Error: input.txt not found!")
        return
    
    lines = [line.strip() for line in lines if line.strip()]
    dataset_num = int(lines[0])
    num_test_cases = int(lines[1])
    
    #load HMM matrices
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
        
        max_obs = len(B[0]) if B and B[0] else 0
        if any(obs >= max_obs for obs in observations):
            print(f"Warning: Some observations are out of range (max allowed: {max_obs - 1})")
            observations = [obs for obs in observations if obs < max_obs]
        
        if not observations:
            print("No valid observations for this test case")
            results.append("")
            continue
        
        try:
            best_states = viterbi(observations, A, B, A[0])
            print(f"Best state sequence: {best_states}")
            if best_states:
                if len(best_states) > 0 and best_states[0] == 0:
                    result_states = best_states[1:] if len(best_states) > 1 else []
                else:
                    result_states = best_states
                
                if result_states:
                    results.append(' '.join(map(str, result_states)))
                else:
                    results.append("0")
            else:
                results.append("0")
                
        except Exception as e:
            print(f"Error processing test case {test_case + 1}: {e}")
            results.append("0")

    try:
        with open("sol-2.txt", 'w') as f:
            for result in results:
                f.write(result + '\n')
    except Exception as e:
        print(f"Error writing results: {e}")

if __name__ == "__main__":
    solve_predictions()
    print("Predictions completed!")