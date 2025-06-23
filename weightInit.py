import numpy as np

def saveMatrix(matrix, file_path):
    with open(file_path, 'w') as file:
        for row in matrix:
            line = ','.join(str(num) for num in row)
            file.write(line + '\n')

def xavier_init(n_in, n_out):
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_out, n_in))

# Xavier initialization
layer_1_matrix = xavier_init(784, 16)
layer_2_matrix = xavier_init(16, 16)
output_layer_matrix = xavier_init(16, 10)

# Initialize all biases to 0
bias_vector_1 = np.zeros((16, 1))
bias_vector_2 = np.zeros((16, 1))
bias_vector_output = np.zeros((10, 1))

# Save to files
saveMatrix(layer_1_matrix, 'layer1Weights.txt')
saveMatrix(layer_2_matrix, 'layer2Weights.txt')
saveMatrix(output_layer_matrix, 'outputLayerWeights.txt')
saveMatrix(bias_vector_1, 'layer1Biases.txt')
saveMatrix(bias_vector_2, 'layer2Biases.txt')
saveMatrix(bias_vector_output, 'outputLayerBiases.txt')
