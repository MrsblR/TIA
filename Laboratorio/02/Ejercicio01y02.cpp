#include <iostream>
#include <vector>
#include <stdexcept> 

float activation(float x) {
    return x >= 0 ? 1.0f : 0.0f;
}

float dotProduct(const std::vector<float>& a, const std::vector<float>& b) {
    if(a.size() != b.size()) {
        throw std::invalid_argument("Error");
    }
    
    float result = 0.0f;
    for(size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

void trainPerceptron(const std::vector<std::vector<float>>& inputs,
                     const std::vector<float>& targets,
                     std::vector<float>& weights,
                     float& bias,
                     float learning_rate,
                     int epochs) {
    if(inputs.empty() || targets.empty()) return;
    if(inputs.size() != targets.size()) {
        throw std::invalid_argument("Error");
    }

    for(int epoch = 0; epoch < epochs; ++epoch) {
        for(size_t i = 0; i < inputs.size(); ++i) {
            // Forward pass
            const float z = dotProduct(weights, inputs[i]) + bias;
            const float prediction = activation(z);
            const float error = targets[i] - prediction;

            // Backward pass (update weights and bias)
            for(size_t j = 0; j < weights.size(); ++j) {
                weights[j] += learning_rate * error * inputs[i][j];
            }
            bias += learning_rate * error;
        }
    }
}

void testPerceptron(const std::vector<std::vector<float>>& inputs,
                    const std::vector<float>& weights,
                    float bias,
                    const std::string& gate_name) {
    std::cout << "\nTesting " << gate_name << " gate:\n";
    for(const auto& input : inputs) {
        const float z = dotProduct(weights, input) + bias;
        std::cout << input[0] << " " << gate_name << " " << input[1]
                  << " = " << activation(z) << "\n";
    }
}

int main() {
    try {
        const std::vector<std::vector<float>> input_AND = {{0,0}, {0,1}, {1,0}, {1,1}};
        const std::vector<float> target_AND = {0, 0, 0, 1};

        const std::vector<std::vector<float>> input_OR = {{0,0}, {0,1}, {1,0}, {1,1}};
        const std::vector<float> target_OR = {0, 1, 1, 1};

        const float learning_rate = 0.1f;
        const int epochs = 100;

        std::vector<float> weights_and(2, 0.0f);
        float bias_and = 0.0f;
        trainPerceptron(input_AND, target_AND, weights_and, bias_and, learning_rate, epochs);
        testPerceptron(input_AND, weights_and, bias_and, "AND");

        std::vector<float> weights_or(2, 0.0f);
        float bias_or = 0.0f;
        trainPerceptron(input_OR, target_OR, weights_or, bias_or, learning_rate, epochs);
        testPerceptron(input_OR, weights_or, bias_or, "OR");

    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}