#pragma once

#include <vector>
#include <random>
#include "Neuron.h"

class NeuralNetwork {
public:
	NeuralNetwork(size_t inputSize, std::vector<int> hiddenLayerSize ,size_t outputSize):
		m_input(inputSize), m_outputSize(outputSize), m_hiddenLayerSize(hiddenLayerSize)
	{
		m_hiddenLayers.reserve(hiddenLayerSize.size());

		size_t prevLayerSize = inputSize;

		for (int m{}; m < hiddenLayerSize.size(); ++m) {

			int layerSize = hiddenLayerSize.at(m);
			m_hiddenLayers.emplace_back();
			auto& layer = m_hiddenLayers.back();
			layer.reserve(layerSize);

			for (int n{}; n < layerSize; ++n) {

				std::vector<float> weight(0,prevLayerSize);
				float bias = 0;

				layer.emplace_back(Neuron(weight, bias));

			}


			prevLayerSize = layerSize;
		}


	}

	size_t m_inputSize;
	size_t m_outputSize;

	std::vector<Neuron> m_output;
	std::vector<std::vector<Neuron>> m_hiddenLayers;
}
