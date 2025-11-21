#pragma once

#include <vector>
#include <random>
#include "Neuron.h"
#include "Math/Math.h"

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

	void SetWeightsAndBiases(
		const std::vector<std::vector<std::vector<float>>>& hiddenWeights,
		const std::vector<std::vector<float>>& hiddenBiases,
		const std::vector<std::vector<float>>& outputWeights,
		const std::vector<float>& outputBiases
	) {
		// Hidden layers
		for (size_t l = 0; l < m_hiddenLayers.size(); ++l) {
			for (size_t n = 0; n < m_hiddenLayers[l].size(); ++n) {
				m_hiddenLayers[l][n].m_weight = hiddenWeights[l][n];
				m_hiddenLayers[l][n].m_bias = hiddenBiases[l][n];
			}
		}

		// Output layer
		for (size_t n = 0; n < m_output.size(); ++n) {
			m_output[n].m_weight = outputWeights[n];
			m_output[n].m_bias = outputBiases[n];
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

	void Train(std::vector<std::vector<float>> datas, std::vector<float> trueValues, float learnRate = 0.1, int epochs = 1000) {

		for (int epoch = 0; epoch < epochs; epoch++) {

			for (int i = 0; i < trueValues.size(); i++) {

				const std::vector<float>& input = datas.at(i);
				float target = trueValues.at(i);

				std::vector<float> outputs = FeedForward(input);

				float predicted = outputs[0];
				float error = predicted - target;


				//Backprop
				std::vector<float> outputDelta(m_outputSize);
				for (size_t j = 0; j < m_output.size(); j++) {
					outputDelta[j] = error * m_output[j].Derivative();
				}

				std::vector<std::vector<float>> hiddenDeltas(m_hiddenLayers.size());

				for (int l = m_hiddenLayers.size() - 1; l >= 0; l--)
				{
					hiddenDeltas[l].resize(m_hiddenLayers[l].size());

					for (size_t n = 0; n < m_hiddenLayers[l].size(); n++)
					{
						float sum = 0.0f;

						// If last hidden layer, use output layer weights
						if (l == m_hiddenLayers.size() - 1) {
							for (size_t k = 0; k < m_outputSize; k++)
								sum += m_output[k].m_weight[n] * outputDelta[k];
						}
						else {
							// Otherwise use next hidden layer weights
							for (size_t k = 0; k < m_hiddenLayers[l + 1].size(); k++)
								sum += m_hiddenLayers[l + 1][k].m_weight[n] * hiddenDeltas[l + 1][k];
						}

						hiddenDeltas[l][n] = sum * m_hiddenLayers[l][n].Derivative();
					}
				}

				// --- UPDATE WEIGHTS ---

				// Update output layer
				for (size_t j = 0; j < m_outputSize; j++) {
					m_output[j].Update(learnRate, outputDelta[j]);
				}
					

				// Update hidden layers
				for (size_t l = 0; l < m_hiddenLayers.size(); l++)
					for (size_t n = 0; n < m_hiddenLayers[l].size(); n++)
						m_hiddenLayers[l][n].Update(learnRate, hiddenDeltas[l][n]);


			}


		}


	}



	size_t m_inputSize;
	size_t m_outputSize;

	std::vector<Neuron> m_output;
	std::vector<std::vector<Neuron>> m_hiddenLayers;
};
