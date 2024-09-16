import numpy as np
import matplotlib.pyplot as plt

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        f.read(16)  
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(-1, 28 * 28) / 255.0 

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        f.read(8) 
        return np.frombuffer(f.read(), dtype=np.uint8)

train_images = load_mnist_images('train-images.idx3-ubyte')
train_labels = load_mnist_labels('train-labels.idx1-ubyte')
test_images = load_mnist_images('t10k-images.idx3-ubyte')
test_labels = load_mnist_labels('t10k-labels.idx1-ubyte')

print("Training set size:", train_images.shape)
print("Test set size:", test_images.shape)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

train_labels_onehot = [vectorized_result(y) for y in train_labels]
training_data = [(np.reshape(x, (784, 1)), vectorized_result(y)) for x, y in zip(train_images, train_labels)]

test_labels_onehot = [vectorized_result(y) for y in test_labels]

class NN:
    def __init__(self, layers):
        self.layers = layers
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def backprop(self, x, y):
        delta_biases = [np.zeros(b.shape) for b in self.biases]
        delta_weights = [np.zeros(w.shape) for w in self.weights]
        
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = (activations[-1] - y) * sigmoid_derivative(zs[-1])
        delta_biases[-1] = delta
        delta_weights[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, len(self.layers)):
            z = zs[-l]
            sp = sigmoid_derivative(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            delta_biases[-l] = delta
            delta_weights[-l] = np.dot(delta, activations[-l - 1].T)
        
        return (delta_biases, delta_weights)

    def update_mini_batch(self, mini_batch, learning_rate):
        delta_biases = [np.zeros(b.shape) for b in self.biases]
        delta_weights = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_b, delta_w = self.backprop(x, y)
            delta_biases = [db + ddb for db, ddb in zip(delta_biases, delta_b)]
            delta_weights = [dw + ddw for dw, ddw in zip(delta_weights, delta_w)]
        self.weights = [w - (learning_rate / len(mini_batch)) * dw
                        for w, dw in zip(self.weights, delta_weights)]
        self.biases = [b - (learning_rate / len(mini_batch)) * db
                       for b, db in zip(self.biases, delta_biases)]

    def train(self, training_data, epochs, mini_batch_size, learning_rate):
        n = len(training_data)
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            print(f'Epoch {epoch+1} complete')

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

nn = NN([784, 30, 10]) 
nn.train(training_data, epochs=10, mini_batch_size=10, learning_rate=3.0)

def evaluate(network, test_images, test_labels):
    predictions = []
    true_labels = []
    for x, y in zip(test_images, test_labels):
        output = network.feedforward(x.reshape(784, 1))
        predicted_label = np.argmax(output)
        predictions.append(predicted_label)
        true_labels.append(y)
    accuracy = sum(int(pred == true) for pred, true in zip(predictions, true_labels)) / len(test_images)
    return accuracy, predictions, true_labels

def show_random_examples(images, predictions, true_labels, num_examples=5):
    indices = np.random.choice(len(images), num_examples, replace=False)
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(images[idx].reshape(28, 28), cmap='gray')
        plt.title(f'Pred: {predictions[idx]}\nTrue: {true_labels[idx]}')
        plt.axis('off')
    plt.show()

accuracy, predictions, true_labels = evaluate(nn, test_images, test_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

show_random_examples(test_images, predictions, true_labels)