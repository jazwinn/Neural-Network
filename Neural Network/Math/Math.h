#pragma once

namespace math
{

	//Activiation function - TODO add the other activations.
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

	float MeanSquaredErrorLoss(const std::vector<float> trueValue, const std::vector<float> predictValue) {
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