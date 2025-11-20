#include <iostream>

#include "NN/Neuron.h"
#include "NN/NeuralNetwork.h"



int main() {
	std::vector<float> x = { 0.9526,0.9526 };
	std::vector<float> wegith = { 0,1 };
	float bias = 0;

	Neuron neuron(wegith, bias);


	std::cout << neuron.FeedForward(x) << std::endl;

    
	return 0;
}