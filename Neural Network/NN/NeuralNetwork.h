#pragma once

#include <vector>
#include <random>
#include "Neuron.h"

class NeuralNetwork {
public:
	NeuralNetwork(size_t inputSize, std::vector<int> hiddenLayerSize, size_t outputSize) :
		m_inputSize(inputSize), m_outputSize(outputSize)
	{
		m_hiddenLayers.reserve(hiddenLayerSize.size());

		size_t prevLayerSize = inputSize;

		for (int m{}; m < hiddenLayerSize.size(); ++m) {

			int layerSize = hiddenLayerSize.at(m);
			m_hiddenLayers.emplace_back();
			auto& layer = m_hiddenLayers.back();
			layer.reserve(layerSize);

			for (int n{}; n < layerSize; ++n) {

				std::vector<float> weight(prevLayerSize, 0.0f);
				float bias = 0;

				layer.emplace_back(Neuron(weight, bias));

			}


			prevLayerSize = layerSize;
		}

		

		for (size_t n{}; n < m_outputSize; ++n) {
			std::vector<float> weight(prevLayerSize, 0.0f);
			float bias = 0;
			m_output.emplace_back(Neuron(weight, bias));
		}


	}

	std::vector<float> FeedForward(const std::vector<float>& input) {
		std::vector<float> layerInput = input;
		std::vector<float> layerOutput;

		// Hidden layers
		for (auto& layer : m_hiddenLayers) {
			layerOutput.clear();
			for (auto& neuron : layer) {
				layerOutput.push_back(neuron.FeedForward(layerInput));
			}
			layerInput = layerOutput; // output becomes input to next layer
		}

		// Output layer
		layerOutput.clear();
		for (auto& neuron : m_output) {
			layerOutput.push_back(neuron.FeedForward(layerInput));
		}

		return layerOutput;
	}

	size_t m_inputSize;
	size_t m_outputSize;

	std::vector<Neuron> m_output;
	std::vector<std::vector<Neuron>> m_hiddenLayers;
};
