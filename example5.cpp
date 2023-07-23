//==================================================================
/// example5.cpp
///
/// Created by Davide Pasca - 2023/07/23
/// See the file "license.txt" that comes with this project for
/// copyright info.
//==================================================================

#include <torch/torch.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>
#include "DrawChart.h"

// sample size in minutes
constexpr size_t SAMPLE_SIZE_MINS = 60*6; // 6 hours

// sine wave parameters for our fictitious market history
constexpr double TRAINDATA_MIN_VAL = 1.0;
constexpr double TRAINDATA_MAX_VAL = 2.0;
constexpr double TRAINDATA_FREQUENCY = 1.0 / 100.0;
// how many minutes of history do we have ?
constexpr size_t TRAINDATA_SAMPLES_MINS_N = 60*24*30*6; // 6 months
// convert the minutes to samples
constexpr size_t TRAINDATA_SAMPLES_N = TRAINDATA_SAMPLES_MINS_N / SAMPLE_SIZE_MINS;

// sine wave parameters for our fictitious market history
constexpr double TESTDATA_MIN_VAL = 1.0;
constexpr double TESTDATA_MAX_VAL = 2.0;
constexpr double TESTDATA_FREQUENCY = 1.0 / 50.0;
// how many minutes of history do we have ?
constexpr size_t TESTDATA_SAMPLES_MINS_N = 60*24*30*1; // 1 month
// convert the minutes to samples
constexpr size_t TESTDATA_SAMPLES_N = TESTDATA_SAMPLES_MINS_N / SAMPLE_SIZE_MINS;

//==================================================================
constexpr size_t LSTM_INPUT_SIZE = 1; // only 1 feature (univariate time series data)
constexpr size_t LSTM_SEQUENCE_LENGTH = TRAINDATA_SAMPLES_N / 20;

// https://www.quora.com/How-should-I-set-the-size-of-hidden-state-vector-in-LSTM-in-keras/answer/Yugandhar-Nanda
constexpr size_t LSTM_HIDDEN_SIZE = LSTM_SEQUENCE_LENGTH * 2;

constexpr double LEARNING_RATE = 0.0001;
constexpr size_t EPOCHS_N = 500;

constexpr size_t CHART_W = 74;
constexpr size_t CHART_H = 12;

//==================================================================
auto generateSineWave(size_t samplesN, double frequency, double minVal, double maxVal)
{
    std::vector<float> sineWave(samplesN);
    double amplitude = (maxVal - minVal) / 2.0;

    for(size_t i=0; i < samplesN; ++i)
        sineWave[i] = (float)(amplitude * (std::sin(2.0 * M_PI * frequency * i) + 1.0) + minVal);

    return sineWave;
}

auto makeLogReturns(const std::vector<float>& vals)
{
    std::vector<float> logReturns(vals.size());

    logReturns[0] = 0.0f; // assume no change in the first element
    for(size_t i=1; i < vals.size(); ++i)
        logReturns[i] = (float)(std::log((double)vals[i]) - std::log((double)vals[i - 1]));

    return torch::tensor(logReturns);
}

//==================================================================
// Function to create sequences
std::pair<torch::Tensor, torch::Tensor> createSequences(const torch::Tensor& data, size_t sequenceLength)
{
    const auto sequencesN = data.size(0) - sequenceLength - 1;
    torch::Tensor sequences = torch::empty({(int64_t)sequencesN, (int64_t)sequenceLength, 1});
    torch::Tensor targets = torch::empty({(int64_t)sequencesN, 1});

    for(int64_t i=0; i < sequencesN; ++i)
    {
        torch::Tensor sequence = data.slice(0, i, i + sequenceLength);
        sequence = sequence.view({(int64_t)sequenceLength, 1});
        sequences[i] = sequence;
        targets[i] = data[i + sequenceLength + 1]; // target is the next value after the sequence
    }

    return std::make_pair(sequences, targets);
}

//==================================================================
class Network : public torch::nn::Module
{
    torch::nn::LSTM mLSTM{nullptr};
    torch::nn::Linear mLinear{nullptr};

public:
    Network(int inputSize, int hiddenSize)
        : mLSTM(inputSize, hiddenSize),
          mLinear(hiddenSize, 1)
    {
        register_module("lstm", mLSTM);
        register_module("linear", mLinear);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        // Initialize hidden and cell states with zeros
        torch::Tensor h0 = torch::zeros({1, x.size(1), mLSTM->options.hidden_size()});
        torch::Tensor c0 = torch::zeros({1, x.size(1), mLSTM->options.hidden_size()});

        // Pass the input through the LSTM layers and output the hidden states
        auto lstm_out = mLSTM->forward(x, std::make_tuple(h0, c0));
        auto out = std::get<0>(lstm_out);

        // Pass the output of the LSTM layer to the linear layer
        // Take only the last output of the LSTM for each sequence
        out = out.slice(1, -1);  // select the last output from each sequence
        out = mLinear->forward(out);

        return out;
    }
};

//==================================================================
void printLossReport(auto epoch, const auto& trainLossHist, const auto& testLossHist)
{
    std::string str;
    str = "## Train Loss\n";
    DrawChart(str, trainLossHist, CHART_W, CHART_H);
    str += "\n";
    str += "## Test Loss\n";
    DrawChart(str, testLossHist, CHART_W, CHART_H);

    std::cout << str << std::endl;

    std::cout << "Epoch [" << (epoch+1) << "/"
        << EPOCHS_N << "], "
        << "Train Loss: " << trainLossHist.back()
        << ", "
        << "Test Loss: " << testLossHist.back() << std::endl;
}

//==================================================================
void plotTargetsAndPredictions(const auto& testTargets, const auto& testPredsVec)
{
    const auto testTargetsCPU = testTargets.cpu();

    std::string str;
    str = "## Test Targets\n";
    DrawChart(str, testTargetsCPU, CHART_W, CHART_H);
    str += "\n";
    str += "## Test Predictions\n";
    DrawChart(str, testPredsVec, CHART_W, CHART_H);

    std::cout << str << std::endl;
}

//==================================================================
int main()
{
    torch::manual_seed(0);

    if (torch::cuda::is_available())
        torch::cuda::manual_seed_all(0);

    //---- SETUP DATA FOR TRAINING
    const auto trainPrices = generateSineWave(
            TRAINDATA_SAMPLES_N,
            TRAINDATA_FREQUENCY,
            TRAINDATA_MIN_VAL,
            TRAINDATA_MAX_VAL);

    const auto trainPriceChangesT = makeLogReturns(trainPrices);

    auto [trainSeqs, trainTargets] = createSequences(trainPriceChangesT, LSTM_SEQUENCE_LENGTH);
    //trainTargets = trainTargets.unsqueeze(1); // add a dimension

    std::cout << __func__ << std::endl;
    std::cout << "  LSTM_INPUT_SIZE: " << LSTM_INPUT_SIZE << std::endl;
    std::cout << "  LSTM_HIDDEN_SIZE: " << LSTM_HIDDEN_SIZE << std::endl;
    std::cout << "  LSTM_SEQUENCE_LENGTH: " << LSTM_SEQUENCE_LENGTH << std::endl;
    std::cout << "  trainSeqs.sizes(): " << trainSeqs.sizes() << std::endl;
    std::cout << "  trainTargets.sizes(): " << trainTargets.sizes() << std::endl;
    std::cout << std::endl;

    // print the data generated so far
    std::string report;

    report += "## Train Prices\n";
    DrawChart(report, trainPrices, CHART_W, CHART_H);
    report += "\n";

    report += "## Train Price Changes\n";
    DrawChart(report, trainPriceChangesT, CHART_W, CHART_H);
    report += "\n";

    std::cout << report << std::endl;

    //---- SETUP DATA FOR TESTING
    const auto testPrices = generateSineWave(
            TESTDATA_SAMPLES_N,
            TESTDATA_FREQUENCY,
            TESTDATA_MIN_VAL,
            TESTDATA_MAX_VAL);

    const auto testPriceChangesT = makeLogReturns(testPrices);

    auto [testSeqs, testTargets] = createSequences(testPriceChangesT, LSTM_SEQUENCE_LENGTH);
    auto testTargetsT = testTargets.unsqueeze(1); // add a dimension

    // Initialize the model
    Network net(LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE);

    // Specify loss function and optimizer
    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(LEARNING_RATE));
    torch::nn::MSELoss criterion{};

    std::vector<float> trainLossHist;
    std::vector<float> testLossHist;

    // Training loop
    for (int epoch = 0; epoch < EPOCHS_N; ++epoch)
    {
        const auto trainPreds = net.forward(trainSeqs).squeeze(-1);
        const auto trainLoss = criterion(trainPreds, trainTargets.view({-1, 1}));

        // Backward pass and optimization
        optimizer.zero_grad();
        trainLoss.backward();
        optimizer.step();

        if ((epoch+1) % 10 == 0)
        {
            // Switch to evaluation mode
            net.eval();

            // Evaluation
            torch::NoGradGuard no_grad;
            const auto testPreds = net.forward(testSeqs).squeeze(-1);
            const auto testLoss = criterion(testPreds, testTargets.view({-1, 1}));

            trainLossHist.push_back(trainLoss.item<float>());
            testLossHist.push_back(testLoss.item<float>());

            //
            std::vector<float> testPredsVec;

            // Iterate over the test sequences one by one
            for (int64_t i=0; i < testSeqs.size(0); ++i)
            {
                // Get the current sequence, add an extra dimension for the batch size
                auto sequence = testSeqs[i].unsqueeze(0);

                // Make a prediction
                auto pred = net.forward(sequence).item<float>();

                // Store the prediction
                testPredsVec.push_back(pred);
            }


            // clear the screen and print the report update
            std::cout << "\033[2J\033[1;1H";

            plotTargetsAndPredictions(testTargets, testPredsVec);

            printLossReport(epoch, trainLossHist, testLossHist);

            // Switch back to training mode
            net.train();
        }
    }
}

