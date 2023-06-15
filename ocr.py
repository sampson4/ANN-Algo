import numpy as np

# Define the Hebb learning function
def hebbian_learning(inputs):
    # Initialize all weights to zero
    weights = np.zeros((35, 35))
    for input in inputs:
        # Update the weights using Hebbian learning rule
        weights += np.outer(input, input)
        return weights
    
# Define the activation function
def activation_function(inputs, weights):
    # Compute the net input
    net_input = np.dot(weights, inputs)
    # Apply the threshold activation function
    output = np.where(net_input >= 0, 1, -1)
    return output

# Define the training inputs and labels
train_inputs = np.array([[1, 1, 1, 1, 1,-1, -1, -1, -1, 1,
-1, -1, -1, -1, 1,
-1, -1, -1, -1, 1,
-1, -1, -1, -1, 1,
-1, -1, -1, -1, 1,
-1, -1, -1, -1, 1],
[-1, -1, 1, -1, -1,
-1, -1, 1, -1, -1,
-1, -1, 1, -1, -1,
-1, -1, 1, -1, -1,
-1, -1, 1, -1, -1,
-1, -1, 1, -1, -1,
-1, -1, 1, -1, -1]])

train_labels = np.array(['H', 'I'])

# Train the network using Hebbian learning
weights = hebbian_learning(train_inputs)


# Define new inputs for recognition
test_inputs = np.array([[-1, 1, 1, 1, -1,
-1, 1, -1, 1, -1,
-1, 1, -1, 1, -1,
-1, 1, -1, 1, -1,
-1, 1, -1, 1, -1,
-1, 1, -1, 1, -1,
-1, 1, -1, 1, -1],
[-1, 1, 1, 1, -1,
-1, -1, 1, -1, -1,
-1, -1, 1, -1, -1,
-1, -1, 1, -1, -1,
-1, -1, 1, -1, -1,
-1, -1, 1, -1, -1,
-1, 1, 1, 1, -1]])

# Recognize the new inputs using the trained network
for test_input in test_inputs:
    output = activation_function(test_input, weights)
    # Find the closest match to a known label
    match_index = np.argmax(np.dot(output, train_inputs.T))
    print("Recognized output: ", train_labels[match_index])

