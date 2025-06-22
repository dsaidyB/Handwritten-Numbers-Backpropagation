import cv2
import numpy as np
import os

images = []
subsetSize = 20
directory = 'MNIST Dataset JPG format/MNIST - JPG - testing'

for i in range(0, 10):
    count = 0
    subdirectory = directory+"/"+str(i)

    for filename in os.listdir(subdirectory):
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
    return np.maximum(0, vector)

def softmax(vector):
    vector = vector.flatten() # Ensure it's 1D
    exps = np.exp(vector - np.max(vector))      # since exponent the relative sizes will be same (can factor out the largest thing), so like can just initially subtract the largest since then have to do smaller exponents
    return exps / np.sum(exps)

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


for i in range(100):
    gradient_list = []
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
        # layer 2 output (16 x 1 matrix)
        layer_2_output = np.matmul(layer_2_matrix, layer_1_processed) - bias_vector_2
        layer_2_processed = relu(layer_2_output)
        # output vector (10 x 1 matrix)
        output_vector = np.matmul(output_layer_matrix, layer_2_processed) - bias_vector_output
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


        # use chain rule relationships to get the gradient for the given image
        bias_vector_output_gradient = []
        output_layer_matrix_gradient = []

        bias_vector_2_gradient = []
        layer_2_matrix_gradient = []

        bias_vector_1_gradient = []
        layer_1_matrix_gradient = []

        for j in range(0, 10):
            dLoss_dZj_L = final_output_vector[j]-true_output_vector[j]
            
            dZj_L_dbj_L = 1
            dLoss_dbj_L = dLoss_dZj_L * dZj_L_dbj_L
            bias_vector_output_gradient.append(dLoss_dbj_L)

            for i in range(0, 16):
                dZj_L_dwji_L = layer_2_processed[i]
                dLoss_dwji_L = dLoss_dZj_L * dZj_L_dwji_L
                output_layer_matrix_gradient.append(dLoss_dwji_L)

                dZj_L_dai_L1 = output_layer_matrix[j][i]
                dLoss_dai_L1 = dLoss_dZj_L * dZj_L_dai_L1

                dai_L1_dZi_L1 = 0
                if (layer_2_output > 0):
                    dai_L1_dZi_L1 = 1
                
                dzi_L1_dbi_L1 = 1
                dLoss_dbi_L1 = dLoss_dai_L1 * dai_L1_dZi_L1 * dzi_L1_dbi_L1
                bias_vector_2_gradient.append(dLoss_dbi_L1)

                for k in range(0, 16):
                    dZi_L1_dwik_L1 = layer_1_processed[k]
                    dLoss_dwik_L1 = dLoss_dai_L1 * dai_L1_dZi_L1 * dZi_L1_dwik_L1
                    layer_2_matrix_gradient.append(dLoss_dwik_L1)

                    dak_L2_dZk_L2 = 0
                    if (layer_1_output > 0):
                        dak_L2_dZk_L2 = 1

                    dZi_L1_dak_L2 = layer_2_matrix[i][k]
                    dLoss_dak_L2 = dLoss_dai_L1 * dai_L1_dZi_L1 * dZi_L1_dak_L2

                    dZk_L2_dbk_L2 = 1
                    dLoss_dbk_L2 = dLoss_dak_L2 * dak_L2_dZk_L2 * dZk_L2_dbk_L2
                    bias_vector_1_gradient.append(dLoss_dbk_L2)

                    for n in range(0, 784):
                        dZk_L2_dwkn_L2 = input_vector[n]
                        dLoss_dwkn_L2 = dLoss_dak_L2 * dak_L2_dZk_L2 * dZk_L2_dwkn_L2
                        layer_1_matrix_gradient.append(dLoss_dwkn_L2)


        gradient_components = bias_vector_1_gradient + bias_vector_2_gradient + bias_vector_output_gradient + layer_1_matrix_gradient + layer_2_matrix_gradient + output_layer_matrix_gradient

        gradient_list.append(np.array(gradient_components))
         

    # then i do this for all images in a training data subset, take the average of them
    gradient_matrix = np.stack(gradient_list)        # Convert list of vectors into a 2D array
    gradient_descent = -1*np.mean(gradient_matrix, axis=0)

    # nudge weights by some scaling factor of negative gradient
    alpha = 0.05

    bias_vector_1 += alpha * gradient_descent[0:16]  # first 16 are for bias vector 1
    bias_vector_2 += alpha * gradient_descent[16:32] # second 16 are for bias vector 2
    bias_vector_output += alpha * gradient_descent[32:42] # third 10 are for bias vector output

    for i in range(len(layer_1_matrix)):
        lower_bound = 42+i*len(layer_1_matrix[i])
        upper_bound = lower_bound + len(layer_1_matrix[i])
        layer_1_matrix[i] += alpha * gradient_descent[lower_bound : upper_bound]

    for j in range(len(layer_2_matrix)):
        lower_bound = 42+len(layer_1_matrix)*len(layer_1_matrix[0])  +  j*len(layer_2_matrix[j])
        upper_bound = lower_bound + len(layer_2_matrix[j])
        layer_2_matrix[j] += alpha * gradient_descent[lower_bound : upper_bound]

    for k in range(len(output_layer_matrix)):
        lower_bound = 42+len(layer_1_matrix)*len(layer_1_matrix[0])+len(layer_2_matrix)*len(layer_2_matrix[0])  +  k*len(output_layer_matrix) 
        upper_bound = lower_bound + len(output_layer_matrix[j])
        output_layer_matrix[k] += alpha * gradient_descent[lower_bound : upper_bound]


saveMatrix(bias_vector_1, 'layer1Biases.txt')
saveMatrix(bias_vector_2, 'layer2Biases.txt')
saveMatrix(bias_vector_output, 'outputLayerBiases.txt')
saveMatrix(layer_1_matrix, 'layer1Weights.txt')
saveMatrix(layer_2_matrix, 'layer2Weights.txt')
saveMatrix(output_layer_matrix, 'outputLayerWeights.txt')
