//==================================================================
/// example2.cpp
///
/// Created by Davide Pasca - 2023/07/23
/// See the file "license.txt" that comes with this project for
/// copyright info.
//==================================================================

#include <torch/torch.h>
#include <iostream>

// Define the network structure
struct Net : torch::nn::Module
{
    Net()
        : fc1(4, 10), // Input layer to hidden layer
        fc2(10, 2)  // Hidden layer to output layer
    {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    torch::nn::Linear fc1, fc2;
};

double calcFitness(torch::Tensor output)
{
    // Your fitness function goes here
    return 0.0;
}

int main()
{
    Net net;

    // Choose a device (CPU or CUDA)
    torch::Device device(torch::kCPU);

    // Move the network to the selected device
    net.to(device);

    // Create a random input tensor
    auto input = torch::randn({1, 4}).to(device);

    // Define a loss function
    torch::nn::MSELoss loss_function;
    loss_function->to(device);

    // Define an optimizer
    torch::optim::SGD optimizer(net.parameters(), /*lr=*/0.01);

    for (size_t epoch = 1; epoch <= 10; ++epoch)
    {
        // Forward pass
        auto output = net.forward(input);

        // Calculate loss
        auto loss = loss_function(output, torch::randn({1, 2}).to(device));

        // Backward pass and optimize
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

#if 0
        // Move output to CPU and convert to double
        auto output_cpu = output.detach().to(torch::kCPU).to(torch::kDouble);

        // Calculate fitness
        double fitness = calcFitness(output_cpu);

        // Print fitness
        std::cout << "Fitness at epoch " << epoch << ": " << fitness << std::endl;
#endif

        // Print loss
        std::cout << "Loss at epoch " << epoch << ": " << loss.item<double>() << std::endl;
    }

    return 0;
}

