//==================================================================
/// example2.cpp
///
/// Created by Davide Pasca - 2023/07/23
/// See the file "license.txt" that comes with this project for
/// copyright info.
//==================================================================

#include <torch/torch.h>
#include <torch/cuda.h>
#include <iostream>

static void test1(c10::DeviceType devType)
{
    auto x = torch::randn({3, 3}, devType);

    torch::Device device = x.device();
    std::cout << "Current device: " << device << std::endl;

    // Perform some operations on x.
    auto y = x + x / 4;

    // Print the result.
    std::cout << y << std::endl;
}

static void test2(c10::DeviceType devType)
{
    auto a = torch::randn({1, 34}, devType);
    a.data_ptr<float>()[0] = 1.0f;
    auto b = torch::randn({1, 34}, devType);
    b.data_ptr<float>()[0] = 2.0f;

    a += b;

    std::cout << a << std::endl;
}

int main()
{
    // print the version of torch
    //std::cout << "Torch version: " << torch::version() << std::endl;
    // print if the GPU is available
    std::cout << "CUDA available: " << torch::cuda::is_available() << std::endl;
    c10::DeviceType devType = torch::kCPU;
         if (torch::cuda::is_available()) devType = torch::kCUDA;
#ifdef __APPLE__
    else if (torch::mps::is_available())  devType = torch::kMPS;
#endif

    std::cout << "Test 1" << std::endl;
    test1(devType);
    std::cout << "Test 2" << std::endl;
    test2(devType);

    std::cout << "Done !" << std::endl;

    return 0;
}
