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

constexpr bool USE_LOG_RETURNS = false;

// sine wave parameters for our fictitious market history
constexpr double TRAINDATA_MIN_VAL = 0.0;
constexpr double TRAINDATA_MAX_VAL = 1.0;
constexpr double TRAINDATA_FREQUENCY = 1.0 / 100.0;
// how many minutes of history do we have ?
constexpr size_t TRAINDATA_SAMPLES_MINS_N = 60*24*30*6; // 6 months
// convert the minutes to samples
constexpr size_t TRAINDATA_SAMPLES_N = TRAINDATA_SAMPLES_MINS_N / SAMPLE_SIZE_MINS;

// sine wave parameters for our fictitious market history
constexpr double TESTDATA_MIN_VAL = 0.0;
constexpr double TESTDATA_MAX_VAL = 1.0;
constexpr double TESTDATA_FREQUENCY = 1.0 / 100.0;
// how many minutes of history do we have ?
constexpr size_t TESTDATA_SAMPLES_MINS_N = 60*24*30*3; // 1 month
// convert the minutes to samples
constexpr size_t TESTDATA_SAMPLES_N = TESTDATA_SAMPLES_MINS_N / SAMPLE_SIZE_MINS;

//==================================================================
constexpr size_t LSTM_INPUT_SIZE = 1; // only 1 feature (univariate time series data)
constexpr size_t LSTM_SEQUENCE_LENGTH = TRAINDATA_SAMPLES_N / 20;
constexpr size_t LSTM_LAYERS_N = 2;

// https://www.quora.com/How-should-I-set-the-size-of-hidden-state-vector-in-LSTM-in-keras/answer/Yugandhar-Nanda
constexpr size_t LSTM_HIDDEN_SIZE = LSTM_SEQUENCE_LENGTH * 2;

constexpr double LEARNING_RATE = 0.0001;
constexpr size_t EPOCHS_N = 10000;

constexpr size_t CHART_W = 74;
constexpr size_t CHART_H = 12;

#if 1
auto normalize(auto val) { return val; }
auto denormalize(auto val) { return val; }
#else
auto normalize(auto val) { return (val + 2) / 4; }
auto denormalize(auto val) { return val * 4 - 2; }
#endif

//==================================================================
auto generateSineWave(size_t samplesN, double frequency, double minVal, double maxVal)
{
    std::vector<float> sineWave(samplesN);
    double amplitude = (maxVal - minVal) / 2.0;

    for(size_t i=0; i < samplesN; ++i)
        sineWave[i] = (float)(amplitude * (std::sin(2.0 * M_PI * frequency * i) + 1.0) + minVal);

    return sineWave;
}

auto makeInputData(const std::vector<float>& vals)
{
    std::vector<float> data(vals.size());

    if (USE_LOG_RETURNS)
    {
        for(size_t i=1; i < vals.size(); ++i)
            data[i] = normalize((float)(std::log((double)vals[i]) - std::log((double)vals[i - 1])) );

        data[0] = data[1];
    }
    else
    {
        for(size_t i=0; i < vals.size(); ++i)
            data[i] = normalize(vals[i]);
    }

    return torch::tensor(data);
}

//==================================================================
// data: {dataN}
// returns: {
//    sequences: {seqs_batch, seq_len, feature},
//      targets: {seqs_batch,          feature}
// }
std::pair<torch::Tensor, torch::Tensor> createSequences(const torch::Tensor& data, int64_t seqLen)
{
    // one sequence for each element in the data, minus one lookback window for the
    //  first prediction to produce, and minus 1 for the prediction itself
    const auto seqsBatchN = data.size(0) - seqLen - 1;
    // sequences: {seqs_batch, seq_len, feature}
    auto sequences = torch::empty({seqsBatchN, seqLen, 1});
    // targets: {seqs_batch, feature}
    auto targets = torch::empty({seqsBatchN, 1});

    for(int64_t i=0; i < seqsBatchN; ++i)
    {
        // sequence: {seq_len}
        const auto sequence = data.slice(0, i, i + seqLen);

        // sequences[i]: {seq_len, feature}
        sequences[i] = sequence.view({seqLen, 1});

        // targets[i]: {feature}
        targets[i] = data[i + seqLen + 1]; // target is the next value after the sequence
    }

    return std::make_pair(sequences, targets);
}

//==================================================================
class Network : public torch::nn::Module
{
    torch::nn::LSTM mLSTM{nullptr};
    torch::nn::Linear mLinear{nullptr};

public:
    Network(int inputSize, int hiddenSize, int layersN)
        : mLSTM(torch::nn::LSTMOptions(inputSize, hiddenSize).num_layers(layersN))
        , mLinear(hiddenSize, 1)
    {
        register_module("lstm", mLSTM);
        register_module("linear", mLinear);
    }

    //       x: {seq_len, seqs_batch, input_size}
    // returns: {seqs_batch}
    torch::Tensor forward(torch::Tensor x)
    {
        // Initialize hidden and cell states with zeros
        const auto& opts = mLSTM->options;
        // h0, c0: {num_layers, seqs_batch, hidden_size}
        torch::Tensor h0 = torch::zeros({opts.num_layers(), x.size(1), opts.hidden_size()});
        torch::Tensor c0 = torch::zeros({opts.num_layers(), x.size(1), opts.hidden_size()});

        // Pass the input through the LSTM layers and output the hidden states
        auto lstm_out = mLSTM->forward(x, std::make_tuple(h0, c0));

        // out: {seq_len, seqs_batch, hidden_size}
        auto out = std::get<0>(lstm_out);

        // Select the last output from each sequence
        // out: {seqs_batch, hidden_size}
        out = out.select(1, -1);  // Select the last element from the seq_len dimension

        // Pass the output of the LSTM layer to the linear layer
        // out: {seqs_batch, output_size}
        out = mLinear->forward(out);

        // Squeeze the last dimension if your output size is 1
        // If output_size was 1, out becomes: {seqs_batch}
        out = out.squeeze(-1);

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
    // Generate a sine wave with given parameters
    // trainPrices: {dataN}
    const auto trainPrices = generateSineWave(
        TRAINDATA_SAMPLES_N, TRAINDATA_FREQUENCY, TRAINDATA_MIN_VAL, TRAINDATA_MAX_VAL);

    // Convert the prices to log returns
    // trainInputData: {dataN}
    const auto trainInputData = makeInputData(trainPrices);

    // Create sequences from the log returns for training
    // trainSeqs:    {seqs_batch, seq_len, feature}
    // trainTargets: {seqs_batch,          feature}
    auto [trainSeqs, trainTargets] = createSequences(trainInputData, LSTM_SEQUENCE_LENGTH);

    std::cout << __func__ << std::endl;
    std::cout << "  LSTM_INPUT_SIZE: " << LSTM_INPUT_SIZE << std::endl;
    std::cout << "  LSTM_HIDDEN_SIZE: " << LSTM_HIDDEN_SIZE << std::endl;
    std::cout << "  LSTM_SEQUENCE_LENGTH: " << LSTM_SEQUENCE_LENGTH << std::endl;
    std::cout << "  trainSeqs    {seqs_batch, seq_len, feature}): "
                        << trainSeqs.sizes() << std::endl;
    std::cout << "  trainTargets {seqs_batch,          feature}): "
                        << trainTargets.sizes() << std::endl;
    std::cout << std::endl;

    // print the data generated so far
    std::string report;

    report += "## Train Prices\n";
    DrawChart(report, trainPrices, CHART_W, CHART_H);
    report += "\n";

    report += "## Train Input Data\n";
    DrawChart(report, trainInputData, CHART_W, CHART_H);
    report += "\n";

    std::cout << report << std::endl;

    //exit(0);

    //---- SETUP DATA FOR TESTING
    // Generate a sine wave with given parameters
    // testPrices: {dataN}
    const auto testPrices = generateSineWave(
        TESTDATA_SAMPLES_N, TESTDATA_FREQUENCY, TESTDATA_MIN_VAL, TESTDATA_MAX_VAL);

    // Convert the prices to log returns
    // testInputData: {dataN}
    const auto testInputData = makeInputData(testPrices);

    // Create sequences from the log returns for testing
    // testSeqs:    {seqs_batch, seq_len, feature}
    // testTargets: {seqs_batch,          feature}
    const auto [testSeqs, testTargets] = createSequences(testInputData, LSTM_SEQUENCE_LENGTH);

    // Initialize the model
    Network net(LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE, LSTM_LAYERS_N);

    // Specify loss function and optimizer
    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(LEARNING_RATE));
    //torch::nn::MSELoss lossFunc(torch::nn::MSELossOptions().reduction(torch::kMean));
    torch::nn::L1Loss lossFunc(torch::nn::L1LossOptions().reduction(torch::kMean));

    std::vector<float> trainLossHist;
    std::vector<float> testLossHist;

    // Training loop
    for (int epoch = 0; epoch < EPOCHS_N; ++epoch)
    {
        // trainPreds: {seqs_batch}
        const auto trainPreds = net.forward(trainSeqs);
        // lossFunc:
        //     trainPreds:                            {seqs_batch}
        //     trainTargets: {seqs_batch, feature} -> {seqs_batch}
        const auto trainLoss = lossFunc(trainPreds, trainTargets.squeeze(-1));

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
            const auto testPreds = net.forward(testSeqs);
            const auto testLoss = lossFunc(testPreds, testTargets.squeeze(-1));

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
                testPredsVec.push_back(denormalize(pred));
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

