# dpasca-pytorch-examples

## Overview

This is just a random testbed/scratchpad for simple experiments with LibTorch, the C++ library on which PyTorch is built.

It's regularly only tested on macOS, but should work on Linux and probably Windows.

Get LibTorch [here](https://pytorch.org/get-started/locally/). Select `LibTorch` or `Source` if you want to build from source.

The CMake build is currently looking for `../libtorch` or `../pytorch-install` (relative to the root of this repo). Change as necessary.

## Build

```bash
./build.sh
```

## Run

```bash
./build/example*
```

## "screenshots"

```
(base) ~/dev/repos/pytorch-mysamples ❯❯❯ ./_build/example5
main
  LSTM_INPUT_SIZE: 1
  LSTM_HIDDEN_SIZE: 72
  LSTM_SEQUENCE_LENGTH: 36
  trainSequencesT.sizes(): [683, 36, 1]
  trainSequencesT.size(0): 683

## Train Prices
 2.1100 |
 2.0236 |    ****                  ****                  ****                  ****
 1.9371 |   *    *                *    *                *    *                *    *
 1.8507 |  *      *              *      *              *      *              *      *
 1.7643 |
 1.6779 | *        *            *        *            *        *            *        *
 1.5914 |
 1.5050 |*          *          *          *          *          *          *          *
 1.4186 |            *        *            *        *            *        *            *
 1.3321 |
 1.2457 |             *      *              *      *              *      *              *
 1.1593 |              *    *                *    *                *    *
 1.0729 |               ****                  ****                  ****
 0.9864 |
 0.9000 |--------------------------------------------------------------------------------

## Train Price Changes
 0.0135 |
 0.0116 |                    ***                   ***                   ***
 0.0097 | *                 *   *                 *   *                 *   *
 0.0078 |  *                     *               *     *               *     *
 0.0059 |   *              *      *                     *                     *
 0.0040 |    *                     *                     *            *        *
 0.0020 |     *           *         *           *         *                     *
 0.0001 |*     *                     *                     *                     *
-0.0018 |       *        *            *        *            *        *            *
-0.0037 |        *                     *                     *                     *
-0.0056 |         *     *               *     *               *     *               *
-0.0076 |          *   *                 *   *                 *   *                 *
-0.0095 |           ***                   ***                   ***                   ***
-0.0114 |
-0.0133 |--------------------------------------------------------------------------------


Epoch [10/500], Train Loss: 5.74979e-05, Test Loss: 0.000261396
Epoch [20/500], Train Loss: 5.67665e-05, Test Loss: 0.000260381
Epoch [30/500], Train Loss: 5.61586e-05, Test Loss: 0.000258909
Epoch [40/500], Train Loss: 5.55857e-05, Test Loss: 0.00025696
Epoch [50/500], Train Loss: 5.49822e-05, Test Loss: 0.000254501
```

