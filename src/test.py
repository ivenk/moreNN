import nn
import mnist_loader

test_data, valid_data, test_data = mnist_loader.load_data_wrapper()

net = nn.Network([784, 30, 10])
net.train(test_data, 30, 30, 3.0, test_data)
