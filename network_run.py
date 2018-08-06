


import mnist_loader
import network
import network2
# ========================================read data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

# ========================================network

# net_1 = network.Network([784, 30, 10])
# net_1.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# ========================================network2

net_2 = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net_2.large_weight_initializer()
net_2.SGD(training_data, 30, 10, 0.1, lmbda = 5.0,evaluation_data=validation_data,
    monitor_evaluation_accuracy=True)

# ========================================network3




