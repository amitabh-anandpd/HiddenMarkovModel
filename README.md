# Hidden Markov Model (HMM)

This section pertains to the Hidden Markov Model. The objective was to construct a model from training data and subsequently utilize it for predictive analysis.

## HMM Construction (`construct.py`)

The primary objective of this task was the construction of the transition probability matrix (A) and the emission probability matrix (B) from provided datasets.

The implementation involves parsing the input data to aggregate frequency counts for state transitions and state-based observations. These counts are then normalized to derive the respective probabilities for the A and B matrices. The finalized matrices are subsequently written to the designated output files.

## Prediction (`predictions.py`)

This task required the prediction of the most probable sequence of hidden states given a sequence of observations, utilizing the previously constructed HMM.

The solution employs the Viterbi algorithm, a dynamic programming approach for finding the most likely sequence of hidden states—known as the Viterbi path. The algorithm iteratively calculates the highest probability path to each state at each time step and stores backpointers. Upon completion of the sequence, the most probable path is reconstructed by backtracking from the final state. The script loads the HMM, executes the algorithm, and outputs the resulting state sequence.