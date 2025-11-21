#include <iostream>

#include "NN/Neuron.h"
#include "NN/NeuralNetwork.h"

int main() {
	//std::vector<float> x = { 0.9526,0.9526 };
	//std::vector<float> wegith = { 0,1 };
	//float bias = 0;

	//Neuron neuron(wegith, bias);


	//std::cout << neuron.FeedForward(x) << std::endl;

	std::vector<std::vector<std::vector<float>>> hiddenWeights = {
		{ {0.f, 1.f}, {0.f, 1.f} }  // first hidden layer
	};
	std::vector<std::vector<float>> hiddenBiases = { {0.f, 0.f}};
	std::vector<std::vector<float>> outputWeights = { {0.f, 1.f}, {0.f, 1.f}};
	std::vector<float> outputBiases = { 0.f };

	NeuralNetwork nn(2, {256,256}, 1);
	
	std::vector<std::vector<float>> data = {
		{-2.f,-1.f}, //Alice
		{25.f,26.f}, //Bob
		{17.f,4.f}, //Charlie
		{-15.f,-6.f}, //Diana
	};

	std::vector<float> truth = {
		1,
		0,
		0,
		1
	};

	nn.Train(data, truth);

	std::vector<float> frank = { 20,2 };
	std::vector<float> emily = { -7,-3 };
	std::vector<float> result = nn.FeedForward(emily);

	for (float f : result) {
		std::cout << f << std::endl;
	}

		
	return 0;
}