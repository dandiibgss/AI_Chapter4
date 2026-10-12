# Exercise 4.1
# Modifikasi Example 4.1 agar neuron melakukan logika OR dari input

from numpy import exp, array, random, dot

#Set up parameters ===============================================
inputs = array([[0, 0], [1, 1], [1, 0], [0, 1]])
outputs = array([[0, 1, 1, 1]]).T      # target OR: 0 OR 0 = 0, selain itu = 1
random.seed(1)
weights = 2 * random.random((2, 1)) -1

#Define a Single Neuron function =================================
def neuron(inputs, weights):

        output = 1 / (1 + exp(-(dot(inputs, weights))))
        return output

#Train the Neuron ================================================
for iteration in range(50000):

        output = 1 / (1 + exp(-(dot(inputs, weights))))
        weights += dot(inputs.T, (outputs -output) * output *
(1 -output))

#Test the Neuron (semua kombinasi input OR) ======================
for x in array([[0, 0], [0, 1], [1, 0], [1, 1]]):
        print(x, "->", neuron(x, weights))
