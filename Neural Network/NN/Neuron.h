#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include "Math/Math.h"

class Neuron {
public:
	Neuron(std::vector<float> weight, float bias):
		m_weight(weight), m_bias(bias)
	{}


	float FeedForward(const std::vector<float>& input) {
		m_input = input;


		float result = 0;
		for (size_t n{}; n < input.size(); n++) {
			result += input[n] * m_weight[n];
		}
		m_result = math::Sigmoid(result + m_bias);
		return m_result;
	}


	float Derivative() {
		return m_result * (1 - m_result);
	}

	void Update(float learningRate, float delta) {
		for (size_t i = 0; i < m_weight.size(); i++)
			m_weight[i] -= learningRate * delta * m_input[i];

		m_bias -= learningRate * delta;
	}

	std::vector<float> m_weight;
	std::vector<float> m_input;
	float m_bias;
	float m_result;
};