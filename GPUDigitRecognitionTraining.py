import cv2
import cupy as cp  # GPU-accelerated numpy-like lib, cupy-cuda12x
import os

print("CUDA devices available:", cp.cuda.runtime.getDeviceCount())
a = cp.array([1, 2, 3])
print("Sample array device:", a.device)  # should print CUDA device info

# 50 subset with 100 epochs --> 5 secs per epoch

# ---------------------- Load Images ----------------------
images = []
subsetSize = 50
directory = 'MNIST Dataset JPG format/MNIST - JPG - training'
fileReadingStartIndex = 250

for i in range(10):
    count = 0
    subdirectory = f"{directory}/{i}"
    files = sorted(os.listdir(subdirectory))  # consistent ordering

    for filename in files[fileReadingStartIndex:]:
        if count >= subsetSize:
            break

        file_path = os.path.join(subdirectory, filename)
        if os.path.isfile(file_path):
            images.append((file_path, i))
            count += 1

# ---------------------- Utility Functions ----------------------
def matrixFromFile(file_path):
    return cp.loadtxt(file_path, delimiter=',')

def saveMatrix(matrix, file_path):
    cp.savetxt(file_path, matrix, delimiter=',')

def relu(x):
    return cp.maximum(0.1 * x, x)  # Leaky ReLU

def softmax(x):
    x = x.flatten()
    x = cp.clip(x - cp.max(x), -200, 200)
    exp_x = cp.exp(x)
    return exp_x / cp.sum(exp_x)

def compute_loss(images, W1, W2, W3, b1, b2, b3):
    loss = 0.0
    eps = 1e-12
    for path, label in images:
        x = cp.array(cv2.imread(path, 0), dtype=cp.float32) / 255.0
        x = x.reshape(-1, 1)

        z1 = W1 @ x - b1
        a1 = relu(z1)
        z2 = W2 @ a1 - b2
        a2 = relu(z2)
        z3 = W3 @ a2 - b3
        y_hat = softmax(z3)

        y = cp.zeros((10, 1))
        y[label] = 1

        loss += -cp.log(y_hat[label] + eps)
    return loss / len(images)

# ---------------------- Load Parameters ----------------------
W1 = matrixFromFile('layer1Weights.txt')
W2 = matrixFromFile('layer2Weights.txt')
W3 = matrixFromFile('outputLayerWeights.txt')

b1 = matrixFromFile('layer1Biases.txt').reshape(-1, 1)  # (16,1)
b2 = matrixFromFile('layer2Biases.txt').reshape(-1, 1)  # (16,1)
b3 = matrixFromFile('outputLayerBiases.txt').reshape(-1, 1)  # (10,1)

# ---------------------- Training Loop ----------------------
all_losses = []
epochs = 100
alpha = 0.01

for epoch in range(epochs):
    grads = cp.zeros(
        b1.size + b2.size + b3.size + W1.size + W2.size + W3.size,
        dtype=cp.float32
    )
    total_loss = 0

    for path, label in images:
        x = cp.array(cv2.imread(path, 0), dtype=cp.float32) / 255.0
        x = x.reshape(-1, 1)

        z1 = W1 @ x - b1
        a1 = relu(z1)
        z2 = W2 @ a1 - b2
        a2 = relu(z2)
        z3 = W3 @ a2 - b3
        y_hat = softmax(z3)

        y = cp.zeros((10, 1))
        y[label] = 1
        loss = -cp.log(y_hat[label] + 1e-12)
        total_loss += loss

        dz3 = y_hat.reshape(-1, 1) - y
        dW3 = dz3 @ a2.T
        db3 = dz3

        da2 = W3.T @ dz3
        dz2 = da2 * (z2 > 0).astype(cp.float32)
        dW2 = dz2 @ a1.T
        db2 = dz2

        da1 = W2.T @ dz2
        dz1 = da1 * (z1 > 0).astype(cp.float32)
        dW1 = dz1 @ x.T
        db1 = dz1

        grad_vec = cp.concatenate([
            db1.flatten(), db2.flatten(), db3.flatten(),
            dW1.flatten(), dW2.flatten(), dW3.flatten()
        ])
        grads += grad_vec

    grads /= len(images)
    gradient_descent = -alpha * grads

    # Unpack gradient and apply update
    offset = 0
    for var in [b1, b2, b3, W1, W2, W3]:
        size = var.size
        var += gradient_descent[offset:offset+size].reshape(var.shape)
        offset += size

    avg_loss = total_loss / len(images)
    all_losses.append(avg_loss)
    print(f"Epoch {epoch+1}: Avg Loss = {avg_loss}")

# ---------------------- Save Results ----------------------
saveMatrix(W1, 'layer1Weights.txt')
saveMatrix(W2, 'layer2Weights.txt')
saveMatrix(W3, 'outputLayerWeights.txt')
saveMatrix(b1, 'layer1Biases.txt')
saveMatrix(b2, 'layer2Biases.txt')
saveMatrix(b3, 'outputLayerBiases.txt')

with open('lossVals.txt', 'a') as f:
    for loss in all_losses:
        f.write(str(float(loss)) + '\n')
