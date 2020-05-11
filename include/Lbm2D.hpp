#pragma once
#include <Common.hpp>
using namespace cl::sycl;

const auto transformWorldCoordinates = [](const float3& worldLocation, const int extentLim = 1, const int lbmSize = 128) {
	return float3{ worldLocation.get_value(X) + extentLim, extentLim - worldLocation.get_value(Y), worldLocation.get_value(Z) } *(lbmSize / (2 * extentLim));
};


const auto densityFunc = [](const float3& lbmSpaceCoordinates)
{
	return 1;
};

// color according to the incoming density
const auto colorFunc = [](const int density)
{
	if (density > 0)
	{
		return float4(0, 0, 1, 1); // blue
	}
	else if (density < 0)
	{
		return float4(1, 1, 0, 1); // yellow
	}
	else
		return  float4(0, 0, 0, 1); // black
};