import cv2
import numpy as np
import os

images = []
subsetSize = 250
directory = 'MNIST Dataset JPG format/MNIST - JPG - training'
fileReadingStartIndex = 250

for i in range(0, 10):
    count = 0
    subdirectory = directory+"/"+str(i)

    for f in range(fileReadingStartIndex, len(os.listdir(subdirectory))):
        filename = os.listdir(subdirectory)[f]

        if (count >= subsetSize):
            break

        else:
            file_path = os.path.join(subdirectory, filename)
            if os.path.isfile(file_path):
                images.append(file_path)
                count += 1

# print(images)


def matrixFromFile(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            cleaned_line = line.strip()     # remove newline characters and extra spaces
            string_values = cleaned_line.split(',')

            row = []
            for value in string_values:
                row.append(float(value))  # convert each string to a float

            matrix.append(row)
    
    return np.array(matrix)

def saveMatrix(matrix, file_path):
    # Write each row to the file, comma-separated
    with open(file_path, 'w') as file:
        for row in matrix:
            line = ','.join(str(num) for num in row)
            file.write(line + '\n')
    # print(f"Matrix saved to {file_path}")

def relu(vector):
    return np.maximum(0.1*vector, vector) # leaky relu

def softmax(vector):
    vector = vector.flatten() # Ensure it's 1D
    shifted_vector = vector - np.max(vector)    # since exponent the relative sizes will be same (can factor out the largest thing), so like can just initially subtract the largest since then have to do smaller exponents
    clipped_vector = np.clip(shifted_vector, -200, 200)  # so the tiny e^-1000 doesnt become zero so it doesnt change that index
    exps = np.exp(clipped_vector)
    return exps / np.sum(exps)

def compute_loss(images, layer_1_matrix, layer_2_matrix, output_layer_matrix,
                 bias_vector_1, bias_vector_2, bias_vector_output):
    total_loss = 0
    eps = 1e-12

    for image in images:
        img = cv2.imread(image, 0)
        normalized_img = np.round(img / 255.0, decimals=5)
        input_vector = normalized_img.flatten().reshape(-1, 1)

        # Forward pass
        layer_1_output = np.matmul(layer_1_matrix, input_vector) - bias_vector_1
        layer_1_processed = relu(layer_1_output)

        layer_2_output = np.matmul(layer_2_matrix, layer_1_processed) - bias_vector_2
        layer_2_processed = relu(layer_2_output)

        output_vector = np.matmul(output_layer_matrix, layer_2_processed) - bias_vector_output
        final_output_vector = softmax(output_vector).flatten()

        # True label
        labelled_number = int(image.split("/")[-1][0])
        true_output_vector = np.zeros(10)
        true_output_vector[labelled_number] = 1

        # Cross-entropy loss
        loss = -np.log(final_output_vector[labelled_number] + eps)
        total_loss += loss

    return total_loss / len(images)


# load weights and biases

# bias vector 1 (16 x 1 matrix)
bias_vector_1 = matrixFromFile('layer1Biases.txt')
# layer 1 (16 x 784 matrix), multiply by input vector, subtract bias, pass this through sigmoid/relu
layer_1_matrix = matrixFromFile('layer1Weights.txt')

# bias vector 2 (16 x 1 matrix)
bias_vector_2 = matrixFromFile('layer2Biases.txt')
# layer 2 (16 x 16 matrix), multiply by new input vector, subtract bias, pass this through sigmoid/relu
layer_2_matrix = matrixFromFile('layer2Weights.txt')

# bias vector 3 (10 x 1 matrix)
bias_vector_output = matrixFromFile('outputLayerBiases.txt')
# output layer (10 x 16 matrix), multiply by new input vector, subtract bias, pass this through sigmoid/relu
output_layer_matrix  = matrixFromFile('outputLayerWeights.txt')

all_losses = []

layer1Outputs = []
layer2Outputs = []
outputVectors = []

epochs = 1

for t in range(epochs):
    gradient_list = []
    total_loss = 0

    for image in images:
        gradient_components = []

        # read image in grayscale mode
        # img = cv2.imread("setupImage.png", 0)
        img = cv2.imread(image, 0) 

        # normalize the img values between 0 and 1
        normalized_img = np.round(img / 255.0, decimals=5)

        # convert the image into a input vector (784 x 1 matrix)
        input_vector = normalized_img.flatten().reshape(-1,1) # -1 is a filler, 1 is saying make it column vector


        # weights say pattern of pixels that is important, bias says how active these pixels need to be
        # layer 1 output (16 x 1 matrix)
        layer_1_output = np.matmul(layer_1_matrix, input_vector) - bias_vector_1
        layer_1_processed = relu(layer_1_output)
        layer1Outputs.append(layer_1_processed)
                             
        # layer 2 output (16 x 1 matrix)
        layer_2_output = np.matmul(layer_2_matrix, layer_1_processed) - bias_vector_2
        layer_2_processed = relu(layer_2_output)
        layer2Outputs.append(layer_2_processed)
                             
        # output vector (10 x 1 matrix)
        output_vector = np.matmul(output_layer_matrix, layer_2_processed) - bias_vector_output
        outputVectors.append(output_vector)
        final_output_vector = softmax(output_vector)
        print(final_output_vector)

        labelled_number = int(image.split("/")[-1][0])
        true_output_vector = np.zeros(10)
        true_output_vector[labelled_number] = 1

        print(labelled_number)
        print(true_output_vector)


        final_output_vector = final_output_vector.flatten()
        eps = 1e-12
        loss = -np.log(final_output_vector[labelled_number] + eps)    # cross-entropy
        print(loss)
        total_loss += loss


        '''
        # use chain rule relationships to get the gradient for the given image
        bias_vector_output_gradient = []
        output_layer_matrix_gradient = []

        bias_vector_2_gradient = []
        layer_2_matrix_gradient = []

        bias_vector_1_gradient = []
        layer_1_matrix_gradient = []

        for j in range(0, 10):
            dLoss_dZj_L = float(final_output_vector[j]-true_output_vector[j])
            # print(dLoss_dZj_L)
            # print(type(dLoss_dZj_L))
            dZj_L_dbj_L = 1
            dLoss_dbj_L = dLoss_dZj_L * dZj_L_dbj_L
            bias_vector_output_gradient.append(float(dLoss_dbj_L))

            for i in range(0, 16):
                dZj_L_dwji_L = float(layer_2_processed[i][0])
                # print(dZj_L_dwji_L)
                # print(type(dZj_L_dwji_L))
                dLoss_dwji_L = dLoss_dZj_L * dZj_L_dwji_L
                output_layer_matrix_gradient.append(float(dLoss_dwji_L))

                dZj_L_dai_L1 = float(output_layer_matrix[j][i])
                # print(type(dZj_L_dai_L1))
                dLoss_dai_L1 = dLoss_dZj_L * dZj_L_dai_L1

                dai_L1_dZi_L1 = 0
                if (float(layer_2_output[i][0]) > 0):  # need this [0] since technically it a 2d array with one column
                    dai_L1_dZi_L1 = 1
                
                dzi_L1_dbi_L1 = 1
                dLoss_dbi_L1 = dLoss_dai_L1 * dai_L1_dZi_L1 * dzi_L1_dbi_L1
                bias_vector_2_gradient.append(float(dLoss_dbi_L1))

                for k in range(0, 16):
                    dZi_L1_dwik_L1 = float(layer_1_processed[k][0])
                    # print(dZi_L1_dwik_L1)
                    dLoss_dwik_L1 = dLoss_dai_L1 * dai_L1_dZi_L1 * dZi_L1_dwik_L1
                    layer_2_matrix_gradient.append(float(dLoss_dwik_L1))

                    dak_L2_dZk_L2 = 0
                    if (float(layer_1_output[k][0]) > 0):
                        dak_L2_dZk_L2 = 1

                    dZi_L1_dak_L2 = float(layer_2_matrix[i][k])
                    # print(dZi_L1_dak_L2)
                    dLoss_dak_L2 = dLoss_dai_L1 * dai_L1_dZi_L1 * dZi_L1_dak_L2

                    dZk_L2_dbk_L2 = 1
                    dLoss_dbk_L2 = dLoss_dak_L2 * dak_L2_dZk_L2 * dZk_L2_dbk_L2
                    bias_vector_1_gradient.append(float(dLoss_dbk_L2))

                    for n in range(0, 784):
                        dZk_L2_dwkn_L2 = float(input_vector[n][0])
                        # print(dZk_L2_dwkn_L2)

                        dLoss_dwkn_L2 = dLoss_dak_L2 * dak_L2_dZk_L2 * dZk_L2_dwkn_L2
                        layer_1_matrix_gradient.append(float(dLoss_dwkn_L2))


        gradient_components = bias_vector_1_gradient + bias_vector_2_gradient + bias_vector_output_gradient + layer_1_matrix_gradient + layer_2_matrix_gradient + output_layer_matrix_gradient
        '''

        # Step 1: output layer delta
        delta_output = final_output_vector.reshape(-1, 1) - true_output_vector.reshape(-1, 1)  # shape (10,1)

        # Step 2: gradient w.r.t output weights and biases
        dL_dW_output = np.dot(delta_output, layer_2_processed.T)  # (10,16)
        dL_db_output = delta_output  # (10,1)

        # Step 3: delta for layer 2
        relu_mask_2 = (layer_2_output > 0).astype(float)
        delta_layer2 = np.dot(output_layer_matrix.T, delta_output) * relu_mask_2  # (16,1)

        # Step 4: gradient w.r.t layer 2 weights and biases
        dL_dW_2 = np.dot(delta_layer2, layer_1_processed.T)  # (16,16)
        dL_db_2 = delta_layer2  # (16,1)

        # Step 5: delta for layer 1
        relu_mask_1 = (layer_1_output > 0).astype(float)
        delta_layer1 = np.dot(layer_2_matrix.T, delta_layer2) * relu_mask_1  # (16,1)

        # Step 6: gradient w.r.t layer 1 weights and biases
        dL_dW_1 = np.dot(delta_layer1, input_vector.T)  # (16,784)
        dL_db_1 = delta_layer1  # (16,1)

        gradient_components = (
            dL_db_1.flatten().tolist() +
            dL_db_2.flatten().tolist() +
            dL_db_output.flatten().tolist() +
            dL_dW_1.flatten().tolist() +
            dL_dW_2.flatten().tolist() +
            dL_dW_output.flatten().tolist()
        )



        gradient_list.append(np.array(gradient_components))
         
    avg_loss = total_loss / len(images)
    print(f"Epoch {t+1}: Average loss = {avg_loss}")
    all_losses.append(avg_loss)

    # then i do this for all images in a training data subset, take the average of them
    gradient_matrix = np.stack(gradient_list)        # Convert list of vectors into a 2D array
    gradient_descent = -1*np.mean(gradient_matrix, axis=0)






    # numerical test

    epsilon = 1e-5
    i, j = 2, 8  # Choose any weight to test
    original_value = layer_1_matrix[i][j]

    # Perturb positively
    layer_1_matrix[i][j] = original_value + epsilon
    loss_plus = compute_loss(images, layer_1_matrix, layer_2_matrix, output_layer_matrix,
                            bias_vector_1, bias_vector_2, bias_vector_output)

    # Perturb negatively
    layer_1_matrix[i][j] = original_value - epsilon
    loss_minus = compute_loss(images, layer_1_matrix, layer_2_matrix, output_layer_matrix,
                            bias_vector_1, bias_vector_2, bias_vector_output)

    # Restore weight
    layer_1_matrix[i][j] = original_value

    # Approximate gradient
    numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)

    # Your actual gradient (get from gradient_descent vector)
    flat_index = i * 784 + j  # because layer_1_matrix is 16 x 784
    bias1_len = bias_vector_1.size
    bias2_len = bias_vector_2.size
    bias3_len = bias_vector_output.size
    index_in_gradient_vector = bias1_len + bias2_len + bias3_len + flat_index
    your_gradient = gradient_descent[index_in_gradient_vector]

    print("Numerical gradient:", numerical_gradient)
    print("Your computed gradient:", your_gradient)





    # nudge weights by some scaling factor of negative gradient
    alpha = 0.01
    print("cycle:", t)

    offset = 0

    # bias_vector_1 update
    size = bias_vector_1.size
    bias_vector_1 += alpha * gradient_descent[offset : offset + size].reshape(bias_vector_1.shape)
    offset += size

    # bias_vector_2 update
    size = bias_vector_2.size
    bias_vector_2 += alpha * gradient_descent[offset : offset + size].reshape(bias_vector_2.shape)
    offset += size

    # bias_vector_output update
    size = bias_vector_output.size
    bias_vector_output += alpha * gradient_descent[offset : offset + size].reshape(bias_vector_output.shape)
    offset += size

    # layer_1_matrix update
    size = layer_1_matrix.size
    layer_1_matrix += alpha * gradient_descent[offset : offset + size].reshape(layer_1_matrix.shape)
    offset += size

    # layer_2_matrix update
    size = layer_2_matrix.size
    layer_2_matrix += alpha * gradient_descent[offset : offset + size].reshape(layer_2_matrix.shape)
    offset += size

    # output_layer_matrix update
    size = output_layer_matrix.size
    output_layer_matrix += alpha * gradient_descent[offset : offset + size].reshape(output_layer_matrix.shape)
    offset += size

    '''
    bias_vector_1 += (alpha * gradient_descent[0:16].reshape(-1, 1))  # first 16 are for bias vector 1
    bias_vector_2 += (alpha * gradient_descent[16:32].reshape(-1, 1)) # second 16 are for bias vector 2
    bias_vector_output += (alpha * gradient_descent[32:42].reshape(-1, 1)) # third 10 are for bias vector output

    for i in range(0, len(layer_1_matrix)):
        lower_bound = 42+i*len(layer_1_matrix[i])
        upper_bound = lower_bound + len(layer_1_matrix[i])
        layer_1_matrix[i] += alpha * gradient_descent[lower_bound : upper_bound]

    for j in range(0, len(layer_2_matrix)):
        lower_bound = 42+len(layer_1_matrix)*len(layer_1_matrix[0])  +  j*len(layer_2_matrix[j])
        upper_bound = lower_bound + len(layer_2_matrix[j])
        layer_2_matrix[j] += alpha * gradient_descent[lower_bound : upper_bound]

    for k in range(0, len(output_layer_matrix)):
        lower_bound = 42+len(layer_1_matrix)*len(layer_1_matrix[0])+len(layer_2_matrix)*len(layer_2_matrix[0])  +  k*len(output_layer_matrix) 
        upper_bound = lower_bound + len(output_layer_matrix[k])
        output_layer_matrix[k] += alpha * gradient_descent[lower_bound : upper_bound]
    '''


print(gradient_descent)
print(all_losses)

saveMatrix(bias_vector_1, 'layer1Biases.txt')
saveMatrix(bias_vector_2, 'layer2Biases.txt')
saveMatrix(bias_vector_output, 'outputLayerBiases.txt')
saveMatrix(layer_1_matrix, 'layer1Weights.txt')
saveMatrix(layer_2_matrix, 'layer2Weights.txt')
saveMatrix(output_layer_matrix, 'outputLayerWeights.txt')


with open('lossVals.txt', 'a') as file:
        for loss in all_losses:
            file.write(str(float(loss)) + '\n')

if (epochs == 1): # only do if we specifically want the max values
    with open('layer1Outputs.txt', 'w') as file:
            for output in layer1Outputs:
                file.write(','.join(map(str, output.flatten())) + '\n')

    layer1Outputs_combined = np.concatenate(layer1Outputs)
    layer1Outputs_max_val = layer1Outputs_combined.max()
    print("layer1Outputs_max_val:", layer1Outputs_max_val)

    with open('layer2Outputs.txt', 'w') as file:
            for output in layer2Outputs:
                file.write(','.join(map(str, output.flatten())) + '\n')
    
    layer2Outputs_combined = np.concatenate(layer2Outputs)
    layer2Outputs_max_val = layer2Outputs_combined.max()
    print("layer2Outputs_max_val:", layer2Outputs_max_val)

    with open('outputVectors.txt', 'w') as file:
            for output in outputVectors:
                file.write(','.join(map(str, output.flatten())) + '\n')
    
    outputVectors_combined = np.concatenate(outputVectors)
    outputVectors_max_val = outputVectors_combined.max()
    print("outputVectors_max_val:", outputVectors_max_val)