#pragma once
#include <vector>
#include <algorithm>
#include <cmath>

class Neuron {
public:
	Neuron(std::vector<float> weight, float bias):
		m_weight(weight), m_bias(bias)
	{}


	float FeedFoward(const std::vector<float>& input) {

		float result = 0;
		for (size_t n{}; n < input.size(); n++) {
			result += input[n] * m_weight[n];
		}
		return Sigmoid(result + m_bias);
	}


	//Activiation function - TODO add the other activations.
	float Sigmoid(float x) {
		return 1 / (1 + std::exp(static_cast<double>(-x)));
	}

	float ReLU(float x) {
		return std::max(0.f,x);
	}


	std::vector<float> m_weight;
	float m_bias;
};