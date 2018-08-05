


import mnist_loader
import network
import network2
# ========================================read data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

# ========================================network

net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# ========================================network2

net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
#net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0,evaluation_data=validation_data,
    monitor_evaluation_accuracy=True)

# ========================================network3




