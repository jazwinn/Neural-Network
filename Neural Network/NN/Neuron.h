#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include "Math/Math.h"

enum ACTIVATION {
	NONE,
	SIGMOID,
	RELU
};

class Neuron {
public:
	Neuron(std::vector<float> weight, float bias):
		m_weight(weight), m_bias(bias)
	{}


	float FeedForward(const std::vector<float>& input, ACTIVATION activation = NONE) {
		m_input = input;


		m_result = 0;
		for (size_t n{}; n < input.size(); n++) {
			m_result += input[n] * m_weight[n];
		}


		switch (activation)
		{
		case SIGMOID:
			m_result = math::Sigmoid(m_result + m_bias);
			break;
		case RELU:
			m_result = math::ReLU(m_result + m_bias);
			break;
		default:
			break;
		}

		


		return m_result;
	}


	float Derivative() {
		return m_result * (1 - m_result);
	}

	void Update(float lr, float delta, const std::vector<float>& inputs)
	{
		for (int i = 0; i < m_weight.size(); i++)
			m_weight[i] -= lr * delta * inputs[i];

		m_bias -= lr * delta;
	}

	std::vector<float> m_weight;
	std::vector<float> m_input;
	float m_bias;
	float m_result;
};