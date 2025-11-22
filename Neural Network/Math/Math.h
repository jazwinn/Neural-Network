#pragma once
#include <algorithm>


namespace math
{

	//Activiation function - TODO add the other activations.

	//Used for binary, multi label?
	float Sigmoid(float x) {
		return 1 / (1 + std::exp(static_cast<double>(-x)));
	}

	float SigmoidDerivative(float x) {

		float sig = Sigmoid(x);
		return sig * (1 - sig);
	}

	float ReLU(float x) {
		return std::max(0.f, x);
	}


	std::vector<float> SoftMax(const std::vector<float>& inputs) {

		if (inputs.empty()) return std::vector<float>{};

		std::vector<float> output(inputs.size());

		//prevents overflow in case where param have too large of a value causing overflow
		float maxValue = *std::max_element(inputs.begin(), inputs.end());

		//sum
		float sum = 0;

		for (int n{}; n < inputs.size(); ++n) {
			output.at(n) = std::exp(inputs.at(n) - maxValue);
			sum += output.at(n);
		}

		//calculate soft max

		for (int n{}; n < inputs.size(); ++n) {
			output.at(n) /= sum;
		}

		return output;
	}

	float MeanSquaredErrorLoss(const std::vector<float>& trueValue, const std::vector<float>& predictValue) {
		size_t total = trueValue.size();

		float sum = 0.0;

		for (size_t n{}; n < total; n++) {
			float tValue = trueValue.at(n);
			float pValue = predictValue.at(n);

			sum += (tValue - pValue) * (tValue - pValue);
		}


		float result = sum / static_cast<float>(total);

		return result;

	}




}