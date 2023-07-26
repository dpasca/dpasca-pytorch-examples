#!/bin/bash

# Create the build directory
mkdir -p _build
cd _build

# Run cmake and build the project
cmake ../cpp -DCMAKE_BUILD_TYPE=Debug && cmake --build . --target all

