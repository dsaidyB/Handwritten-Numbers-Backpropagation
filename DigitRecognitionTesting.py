import cv2
import numpy as np
import os

images = []
subsetSize = 1
directory = 'MNIST Dataset JPG format/MNIST - JPG - testing'
fileReadingStartIndex = 800

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

def relu(vector):
    return np.maximum(0.1*vector, vector) # leaky relu

def softmax(vector):
    vector = vector.flatten() # Ensure it's 1D
    shifted_vector = vector - np.max(vector)    # since exponent the relative sizes will be same (can factor out the largest thing), so like can just initially subtract the largest since then have to do smaller exponents
    clipped_vector = np.clip(shifted_vector, -200, 200)  # so the tiny e^-1000 doesnt become zero so it doesnt change that index
    exps = np.exp(clipped_vector)
    return exps / np.sum(exps)


# load weights and biases
bias_vector_1 = matrixFromFile('layer1Biases.txt')
layer_1_matrix = matrixFromFile('layer1Weights.txt')
bias_vector_2 = matrixFromFile('layer2Biases.txt')
layer_2_matrix = matrixFromFile('layer2Weights.txt')
bias_vector_output = matrixFromFile('outputLayerBiases.txt')
output_layer_matrix  = matrixFromFile('outputLayerWeights.txt')

correct = 0
numTotalImages = len(images)

for image in images:
    # read image in grayscale mode
    img = cv2.imread(image, 0) 
    # normalize the img values between 0 and 1
    normalized_img = np.round(img / 255.0, decimals=5)
    # convert the image into a input vector (784 x 1 matrix)
    input_vector = normalized_img.flatten().reshape(-1,1)
    print(input_vector)

    # passing input vector through trained model
    layer_1_output = np.matmul(layer_1_matrix, input_vector) - bias_vector_1
    layer_1_processed = relu(layer_1_output)
    layer_2_output = np.matmul(layer_2_matrix, layer_1_processed) - bias_vector_2
    layer_2_processed = relu(layer_2_output)
    output_vector = np.matmul(output_layer_matrix, layer_2_processed) - bias_vector_output
    final_output_vector = softmax(output_vector)
    print(final_output_vector)
    predictedDigit = np.argmax(final_output_vector)
    print("Predicted Digit:", predictedDigit)

    # get image label
    labelled_number = int(image.split("/")[-1][0])
    print("True Digit:", labelled_number)
    true_output_vector = np.zeros(10)
    true_output_vector[labelled_number] = 1
    # print(true_output_vector)

    if (predictedDigit == labelled_number):
        correct += 1

modelAccuracy = correct/numTotalImages * 100.0
print("Accuracy: " + str(modelAccuracy) + "%")
