#pragma once
#include <Common.hpp>
using namespace cl::sycl;

struct SphericalCoordinates {
	float r = 0;
	float theta = 0;
	float phi = 0;

};

const auto Y_l1_m0 = [](const SphericalCoordinates &sphericalCoordinates) {
	return 1.0f / 2.0f * float(cl::sycl::sqrt(3.0f / M_PI)) * float(cl::sycl::cos(sphericalCoordinates.theta));
};

const auto Y_l2_m0 = [](const SphericalCoordinates &sphericalCoordinates) {
	return 1.0f / 4.0f * float(cl::sycl::sqrt(5.0f / M_PI)) * 
		(3.f * cl::sycl::pow(float(cl::sycl::cos(sphericalCoordinates.theta)), 2.f) - 1);
};

const auto Y_l3_m0 = [](const SphericalCoordinates &sphericalCoordinates) {
	return 1.0f / 4.0f * float(cl::sycl::sqrt(7.0f / M_PI)) * 
		(5.f * cl::sycl::pow(float(cl::sycl::cos(sphericalCoordinates.theta)), 3.f) 
			- 3.f * float(cl::sycl::cos(sphericalCoordinates.theta)));
};

const auto densityFunc = [](const SphericalCoordinates &sphericalCoordinates)
{
	float val = Y_l1_m0(sphericalCoordinates);
	float result = cl::sycl::fabs(2 * cl::sycl::fabs(val) - sphericalCoordinates.r);
	const float thickness = 0.4f;

	if (result < thickness)	// thickness of shell 
		return val < 0 ? -1 : 1;
	else
		return 0;
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

const auto transformWorldCoordinates = [](const float3& location) {
	auto r = cl::sycl::length(location);
	return SphericalCoordinates{ r,  cl::sycl::acos(location.get_value(Z) / r), cl::sycl::atan2(location.get_value(Y), location.get_value(X)) };
};