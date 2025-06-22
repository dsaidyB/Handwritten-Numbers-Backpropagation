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


# read image in grayscale mode
# img = cv2.imread("setupImage.png", 0)

for image in images:
    img = cv2.imread(image, 0) 

    # normalize the img values between 0 and 1
    normalized_img = np.round(img / 255.0, decimals=5)

    # convert the image into a input vector (784 x 1 matrix)
    input_vector = normalized_img.flatten().reshape(-1,1) # -1 is a filler, 1 is saying make it column vector


    # weights say pattern of pixels that is important, bias says how active these pixels need to be


    # bias vector 1 (16 x 1 matrix)
    bias_vector_1 = matrixFromFile('layer1Biases.txt')
    # layer 1 (16 x 784 matrix), multiply by input vector, subtract bias, pass this through sigmoid/relu
    layer_1_matrix = np.random.rand(16, 784)  # Values from 0 to 1
    # layer 1 output (16 x 1 matrix)
    layer_1_output = np.matmul(layer_1_matrix, input_vector) - bias_vector_1


    # bias vector 2
    bias_vector_2 = np.random.rand(16, 1)  # Values from 0 to 1
    # layer 2 (16 x 16 matrix), multiply by new input vector, subtract bias, pass this through sigmoid/relu
    layer_2_matrix = np.random.rand(16, 16)
    # layer 2 output (16 x 1 matrix)
    layer_2_output = np.matmul(layer_2_matrix, layer_1_output) - bias_vector_2

    # bias vector 3
    bias_vector_output = np.random.rand(10, 1)  # Values from 0 to 1
    # output layer (10 x 16 matrix), multiply by new input vector, subtract bias, pass this through sigmoid/relu
    output_layer_matrix  = np.random.rand(10, 16)  # Values from 0 to 1
    # output vector (10 x 1 matrix)
    output_vector = np.matmul(output_layer_matrix, layer_2_output) - bias_vector_output

    print(output_vector)


    # ok so then i need to subtract from desired value, take squared error
    # use chain rule relationships to get the gradient for the given image

    # then i do this for all images in a training data subset, take the average of them

    # nudge weights by some scaling factor of negative gradient 


