//==================================================================
/// example3.cpp
///
/// Created by Davide Pasca - 2023/07/23
/// See the file "license.txt" that comes with this project for
/// copyright info.
//==================================================================

#include <torch/torch.h>

// Define a few weights and biases
torch::Tensor weight1 = torch::randn({2, 1});
torch::Tensor bias1   = torch::randn({2});
torch::Tensor weight2 = torch::randn({3, 2});
torch::Tensor bias2   = torch::randn({3});
torch::Tensor weight3 = torch::randn({4, 3});
torch::Tensor bias3   = torch::randn({4});

torch::Tensor forwardCascade(torch::Tensor input, torch::nn::Linear& layer1, torch::nn::Linear& layer2, torch::nn::Linear& layer3) {
    torch::Tensor output1 = layer1->forward(input);
    torch::Tensor output2 = layer2->forward(output1);
    torch::Tensor output3 = layer3->forward(output2);
    return output3;
}

torch::Tensor bmmCascade(torch::Tensor input, torch::Tensor weight1, torch::Tensor bias1, torch::Tensor weight2, torch::Tensor bias2, torch::Tensor weight3, torch::Tensor bias3) {
    torch::Tensor output1 = torch::bmm(input.unsqueeze(0),   weight1.t().unsqueeze(0)).squeeze(0) + bias1;
    torch::Tensor output2 = torch::bmm(output1.unsqueeze(0), weight2.t().unsqueeze(0)).squeeze(0) + bias2;
    torch::Tensor output3 = torch::bmm(output2.unsqueeze(0), weight3.t().unsqueeze(0)).squeeze(0) + bias3;
    return output3;
}

int main()
{
    // Define the layers
    torch::nn::Linear layer1(weight1.size(0), weight1.size(1));
    torch::nn::Linear layer2(weight2.size(0), weight2.size(1));
    torch::nn::Linear layer3(weight3.size(0), weight3.size(1));

    // Manually set the weights and biases of the layers
    layer1->weight = weight1;
    layer1->bias = bias1;
    layer2->weight = weight2;
    layer2->bias = bias2;
    layer3->weight = weight3;
    layer3->bias = bias3;

    // Define an input
    torch::Tensor input = torch::randn({5, 1});

    // Perform the forward cascade
    torch::Tensor outputForward = forwardCascade(input, layer1, layer2, layer3);

    // Perform the bmm cascade
    torch::Tensor outputBmm = bmmCascade(input, weight1, bias1, weight2, bias2, weight3, bias3);

    // Print the output tensors
    std::cout << "Output (Forward): " << outputForward << std::endl;
    std::cout << "Output (BMM): " << outputBmm << std::endl;

    return 0;
}
