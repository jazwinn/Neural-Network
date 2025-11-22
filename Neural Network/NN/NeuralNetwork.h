#pragma once

#include <vector>
#include <random>
#include "Neuron.h"
#include "Math/Math.h"

class NeuralNetwork {

	struct FowardCache {
		std::vector<std::vector<float>> hiddenInputs;
		std::vector<std::vector<float>> hiddenOutputs;
		std::vector<float> outputInputs;
		std::vector<float> outputSoftmax;
	};

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

			std::random_device rd;
			std::mt19937 gen(rd());
			float limit = std::sqrt(6.0f / (prevLayerSize + layerSize)); // Glorot
			std::uniform_real_distribution<float> dist(-limit, limit);

			for (int n{}; n < layerSize; ++n) {

				std::vector<float> weight(prevLayerSize);
				for (size_t w = 0; w < prevLayerSize; ++w)
					weight[w] = dist(gen);


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
				layerOutput.push_back(neuron.FeedForward(layerInput, SIGMOID));
			}
			layerInput = layerOutput; // output becomes input to next layer
		}

		// Output layer
		layerOutput.clear();
		for (auto& neuron : m_output) {
			layerOutput.push_back(neuron.FeedForward(layerInput));
		}

		return math::SoftMax(layerOutput);
	}

	FowardCache FeedForwardCached(const std::vector<float>& input) {
		FowardCache cache;
		std::vector<float> layerInput = input;
		std::vector<float> layerOutput;

		// Hidden layers
		for (auto& layer : m_hiddenLayers) {

			cache.hiddenInputs.push_back(layerInput);

			layerOutput.clear();
			for (auto& neuron : layer) {
				layerOutput.push_back(neuron.FeedForward(layerInput, SIGMOID));
			}

			cache.hiddenOutputs.push_back(layerOutput);

			layerInput = layerOutput; // output becomes input to next layer
		}

		// Output layer
		cache.outputInputs.clear();
		for (auto& neuron : m_output) {
			cache.outputInputs.push_back(neuron.FeedForward(layerInput));
		}

		cache.outputSoftmax = math::SoftMax(cache.outputInputs);

		return cache;
	}

	//trueValue must be one hot sth sth sth e.g. [1,0,0] or [0,1,0]
	void Train(const std::vector<std::vector<float>>& datas, const std::vector<int>& trueValues, int numClass,float learnRate = 0.1, int epochs = 1000) {

		std::vector<std::vector<float>> targestOneHot = OneHotEncode(trueValues, numClass);


		float loss = 0.0f;

		for (int epoch = 0; epoch < epochs; epoch++) {

			for (int i = 0; i < trueValues.size(); i++) {

				const std::vector<float>& input = datas.at(i);
				FowardCache cache = FeedForwardCached(input);
				std::vector<float> outputs = cache.outputSoftmax;


				//logs
				for (size_t j = 0; j < m_outputSize; ++j)
					if (targestOneHot[i][j] > 0.5f)
						loss += -std::log(outputs[j] + 1e-7f);
				if (!(epoch % 10)) {
					std::cout << "Average loss after 10 iteration" << loss / 10.f << std::endl;
					loss = 0.f;
				}




				//Backprop
				std::vector<float> outputDelta(m_outputSize);
				for (size_t j = 0; j < m_output.size(); j++) {

					float error = outputs.at(j) - targestOneHot.at(i).at(j);
					//outputDelta[j] = error * m_output[j].Derivative();
					outputDelta[j] = error;
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
				const std::vector<float>& lastHiddenOutput = cache.hiddenOutputs.back();
				for (size_t j = 0; j < m_outputSize; j++) {
					m_output[j].Update(learnRate, outputDelta[j], lastHiddenOutput);
				}
					

				// Update hidden layers
				for (size_t l = 0; l < m_hiddenLayers.size(); l++)
				{
					const std::vector<float>& prevInputs =
						(l == 0 ? input : cache.hiddenOutputs[l - 1]);

					for (size_t n = 0; n < m_hiddenLayers[l].size(); n++)
						m_hiddenLayers[l][n].Update(
							learnRate,
							hiddenDeltas[l][n],
							prevInputs
						);
				}
					


			}


		}


	}

	std::vector<std::vector<float>> OneHotEncode(const std::vector<int>& labels, int numClasses)
	{
		std::vector<std::vector<float>> oneHot(labels.size(), std::vector<float>(numClasses, 0.0f));

		for (size_t i = 0; i < labels.size(); i++)
		{
			int label = labels[i];
			if (label >= 0 && label < numClasses)
				oneHot[i][label] = 1.0f;
		}

		return oneHot;
	}

	size_t m_inputSize;
	size_t m_outputSize;

	std::vector<Neuron> m_output;
	std::vector<std::vector<Neuron>> m_hiddenLayers;
};
