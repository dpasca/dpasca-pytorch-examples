//==================================================================
/// DrawChart.h
///
/// Created by Davide Pasca - 2023/07/23
/// See the file "license.txt" that comes with this project for
/// copyright info.
//==================================================================

#ifndef DRAWCHART_H
#define DRAWCHART_H

#include <vector>
#include <algorithm>
#include <string>
#include <limits>

//==================================================================
inline void DrawChart(
    std::string& outStr,
    const auto* pVals,
    size_t valsN,
    size_t dispW,
    size_t dispH,
    double baseY=std::numeric_limits<double>::max())
{
    const auto [minIt, maxIt] = std::minmax_element(pVals, pVals + valsN);
    auto mi = *minIt;
    auto ma = *maxIt;
    mi -= (ma - mi) * 0.1; // add 10% padding
    ma += (ma - mi) * 0.1;

    std::vector<std::vector<char>> buf(dispH, std::vector<char>(dispW, ' '));

    auto valToScreenY = [&](double v) -> size_t
    {
        return dispH - 1 - (v - mi) / (ma - mi) * (dispH - 1);
    };
    auto screenYToVal = [&](size_t y) -> double
    {
        return mi + (ma - mi) * (dispH - 1 - y) / (dispH - 1);
    };
    auto screenXToValIdx = [&](size_t x) -> size_t
    {
        return std::clamp<size_t>((double)x * (valsN - 1) / (dispW - 1), 0, valsN-1);
    };

    std::vector<std::string> leftRules(dispH);
    for (size_t i=0; i < dispH; ++i)
    {
        char buff[128] {};
        snprintf(buff, sizeof(buff), "% 10.7f |", screenYToVal(i));
        leftRules[i] = buff;
    }

    for (size_t i=0; i < dispW; ++i) // borders
        buf[dispH - 1][i] = '-';

    if (baseY != std::numeric_limits<double>::max()) // dotted line at baseY
    {
        const auto y = valToScreenY(baseY);
        for (size_t i=0; i < dispW; ++i)
            buf[y][i] = '.';
    }

    for (size_t i=0; i < dispW; ++i) // plot
    {
        const auto y = valToScreenY(pVals[ screenXToValIdx(i) ]);
        buf[y][i] = '*';
    }

    for (size_t i=0; i < dispH; ++i) // print
    {
        outStr += leftRules[i];
        for (size_t j=0; j < dispW; ++j)
            outStr += buf[i][j];
        outStr += '\n';
    }
}

inline void DrawChart(
    std::string& outStr,
    const std::vector<float>& vals,
    size_t dispW,
    size_t dispH,
    double baseY=std::numeric_limits<double>::max())
{
    DrawChart(outStr, vals.data(), vals.size(), dispW, dispH, baseY);
}

inline void DrawChart(
    std::string& outStr,
    const torch::Tensor& vals,
    size_t dispW,
    size_t dispH,
    double baseY=std::numeric_limits<double>::max())
{
    DrawChart(outStr, vals.data_ptr<float>(), vals.numel(), dispW, dispH, baseY);
}

#endif

