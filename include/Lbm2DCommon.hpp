#pragma once

struct Distributions {
	float f0;
	cl::sycl::float4 f1234;
	cl::sycl::float4 f5678;
};

struct CellData
{
	Distributions distributions;
	cl::sycl::float2 velocity;
	bool cellType;

};

const auto colorFunc = [](cl::sycl::float2 inVelocity, bool isBoundary) {
	using namespace cl::sycl;
	float4 color = { 0.f, 0.f, 0.f, 1.f };

	// creat a color scale (use 4th value now for magnitude, later set the alpha channel here)
	float4 color1{ 0, 0, 0, 0.0 };
	float4 color2{ 0, 0, 1, 0.2 };
	float4 color3{ 0, 1, 1, 0.4 };
	float4 color4{ 0, 1, 0, 0.8 };
	float4 color5{ 1, 1, 0, 1.6 };
	float4 color6{ 1, 0, 0, 3.2 };

	if (isBoundary) {
		color = { 0.f, 0.f, 0.f, 1.f };
	}
	else {
		auto velocityMangitude = cl::sycl::length(inVelocity) * 20;

		int i = 0;
		float w;

		if (velocityMangitude <= color1.get_value(3))
		{
			color = color1;
		}
		else if (velocityMangitude >= color6.get_value(3))
		{
			color = color6;
		}
		else
		{
			float4 colorBoundaryStart;
			float4 colorBoundaryEnd;
			if ((float)color1.get_value(3) <= velocityMangitude && velocityMangitude < color2.get_value(3)) {
				colorBoundaryStart = color1;
				colorBoundaryEnd = color2;
			}
			else if ((float)color2.get_value(3) <= velocityMangitude && velocityMangitude < color3.get_value(3)) {
				colorBoundaryStart = color2;
				colorBoundaryEnd = color3;

			}
			else if ((float)color3.get_value(3) <= velocityMangitude && velocityMangitude < color4.get_value(3)) {
				colorBoundaryStart = color3;
				colorBoundaryEnd = color4;
			}
			else if ((float)color4.get_value(3) <= velocityMangitude && velocityMangitude < color5.get_value(3)) {
				colorBoundaryStart = color4;
				colorBoundaryEnd = color5;
			}
			else if ((float)color5.get_value(3) <= velocityMangitude && velocityMangitude < color6.get_value(3)) {
				colorBoundaryStart = color5;
				colorBoundaryEnd = color6;
			}

			// linear interpolation
			w = (velocityMangitude - colorBoundaryStart.get_value(3)) / (colorBoundaryEnd.get_value(3) - colorBoundaryStart.get_value(3));
			color = (1 - w) * colorBoundaryStart + w * colorBoundaryEnd;
		}
	}
	// set alpha to 1;
	color.set_value(3, 1.f);

	return color;
};


const auto computefEq = [](float rho, float weight, cl::sycl::float2 dir, cl::sycl::float2 velocity) {

	float u2 = cl::sycl::dot(velocity, velocity);
	float eu = cl::sycl::dot(dir, velocity);
	return rho * weight * (1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * u2);
};


const auto collide = [](const Distributions& cellDistributions, const bool cellType) {
	using namespace cl::sycl;
	//qDebug() << x << " " << y << " " << pos << "\n";
			// Read input distributions
	float f0 = cellDistributions.f0;
	float4 f1234 = cellDistributions.f1234;
	float4 f5678 = cellDistributions.f5678;
	bool type = cellType;

	const int dirX[9] = { 0, 1, 0, -1,  0, 1, -1,  -1,  1 };
	const int dirY[9] = { 0, 0, 1,  0, -1, 1,  1,  -1, -1 };

	const float weight[9] = { 4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0 };

	// (1/relaxation time) Related to viscosity 
	float omega = 1.2f;

	float rho;
	float2 u;

	//boundary
	if (type) {
		// Swap directions by swizzling // Ez igy nem jo, de ez csak a hataroknal jelent problemat
		/*f1234.x() = f1234.z();
		f1234.y() = f1234.w();
		f1234.z() = f1234.x();
		f1234.w() = f1234.y();

		f5678.x() = f5678.z();
		f5678.y() = f5678.w();
		f5678.z() = f5678.x();
		f5678.w() = f5678.y();*/

		rho = 0;
		u = float2{ 0.f, 0.f };
	}
	// fluid
	else
	{
		// Compute rho and u
		// Rho is computed by doing a reduction on f
		rho = f0 + f1234.get_value(0) + f1234.get_value(1) + f1234.get_value(2) + f1234.get_value(3)
			+ f5678.get_value(0) + f5678.get_value(1) + f5678.get_value(2) + f5678.get_value(3);

		// Compute velocity

		/*qDebug() << dirY[1] << " " << dirY[2] << " " << dirY[3] << " " << dirY[4] << " " << dirY[5] << " " << dirY[6] << " " << dirY[7] << " " << dirY[8] << "\n";
		qDebug() << dirX[1] << " " << dirX[2] << " " << dirX[3] << " " << dirX[4] << " " << dirX[5] << " " << dirX[6] << " " << dirX[7] << " " << dirX[8] << "\n";*/


		float uX = (f1234.get_value(0) * dirX[1] + f1234.get_value(1) * dirX[2] + f1234.get_value(2) * dirX[3] + f1234.get_value(3) * dirX[4]
			+ f5678.get_value(0) * dirX[5] + f5678.get_value(1) * dirX[6] + f5678.get_value(2) * dirX[7] + f5678.get_value(3) * dirX[8]) / rho;

		u.set_value(0, uX);

		float uY = (f1234.get_value(0) * dirY[1] + f1234.get_value(1) * dirY[2] + f1234.get_value(2) * dirY[3] + f1234.get_value(3) * dirY[4]
			+ f5678.get_value(0) * dirY[5] + f5678.get_value(1) * dirY[6] + f5678.get_value(2) * dirY[7] + f5678.get_value(3) * dirY[8]) / rho;

		u.set_value(1, uY);


		float4 fEq1234;	// Stores feq 
		float4 fEq5678;
		float fEq0;

		if (uX > 0 || uY > 0) {
			int breakpointHere = 1;
		}

		//Compute fEq
		fEq0 = computefEq(rho, weight[0], float2{ 0, 0 }, u);
		fEq1234.set_value(0, computefEq(rho, weight[1], float2{ dirX[1], dirY[1] }, u));
		fEq1234.set_value(1, computefEq(rho, weight[2], float2{ dirX[2], dirY[2] }, u));
		fEq1234.set_value(2, computefEq(rho, weight[3], float2{ dirX[3], dirY[3] }, u));
		fEq1234.set_value(3, computefEq(rho, weight[4], float2{ dirX[4], dirY[4] }, u));

		fEq5678.set_value(0, computefEq(rho, weight[5], float2{ dirX[5], dirY[5] }, u));
		fEq5678.set_value(1, computefEq(rho, weight[6], float2{ dirX[6], dirY[6] }, u));
		fEq5678.set_value(2, computefEq(rho, weight[7], float2{ dirX[7], dirY[7] }, u));
		fEq5678.set_value(3, computefEq(rho, weight[8], float2{ dirX[8], dirY[8] }, u));

		f0 = (1 - omega) * f0 + omega * fEq0;
		f1234 = (1 - omega) * f1234 + omega * fEq1234;
		f5678 = (1 - omega) * f5678 + omega * fEq5678;
	}

	return CellData{ Distributions{f0, f1234, f5678}, u, type };
};

template <cl::sycl::access::target Target>
const auto streamToNeighbours = [](const cl::sycl::int2 id, const int currentPos, const int width, const int height, const Distributions& currentCellDistributions,
	const cl::sycl::accessor<float, 1, cl::sycl::access::mode::discard_write, Target, cl::sycl::access::placeholder::false_t>& of0,
	const cl::sycl::accessor<cl::sycl::float4, 1, cl::sycl::access::mode::discard_write, Target, cl::sycl::access::placeholder::false_t>& of1234,
	const cl::sycl::accessor<cl::sycl::float4, 1, cl::sycl::access::mode::discard_write, Target, cl::sycl::access::placeholder::false_t>& of5678) {

	using namespace cl::sycl;
	// Propagate
	// New positions to write (Each thread will write 8 values)

	const int dirX[9] = { 0, 1, 0, -1,  0, 1, -1,  -1,  1 };
	const int dirY[9] = { 0, 0, 1,  0, -1, 1,  1,  -1, -1 };

	int8 x8 = int8(id.get_value(0));
	int8 y8 = int8(id.get_value(1));
	int8 width8 = int8(width);

	int8 nX = x8 + int8(dirX[1], dirX[2], dirX[3], dirX[4], dirX[5], dirX[6], dirX[7], dirX[8]);
	int8 nY = y8 + int8(dirY[1], dirY[2], dirY[3], dirY[4], dirY[5], dirY[6], dirY[7], dirY[8]);
	int8 nPos = nX + width8 * nY;


	int isNotRightBoundary = id.get_value(0) < int(width - 1); // Not on Right boundary
	int isNotUpperBoundary = id.get_value(1) > int(0);                      // Not on Upper boundary
	int isNotLeftBoundary = id.get_value(0) > int(0);                      // Not on Left boundary
	int isNotLowerBoundary = id.get_value(1) < int(height - 1); // Not on lower boundary

	of0[currentPos] = currentCellDistributions.f0;

	// Propagate to right cell
	if (isNotRightBoundary) {
		of1234[nPos.get_value(0)].set_value(0, currentCellDistributions.f1234.get_value(0));
	}

	// Propagate to Lower cell
	if (isNotLowerBoundary) {
		of1234[nPos.get_value(1)].set_value(1, currentCellDistributions.f1234.get_value(1));
	}

	// Propagate to left cell
	if (isNotLeftBoundary) {
		of1234[nPos.get_value(2)].set_value(2, currentCellDistributions.f1234.get_value(2));
	}

	// Propagate to Upper cell
	if (isNotUpperBoundary) {
		of1234[nPos.get_value(3)].set_value(3, currentCellDistributions.f1234.get_value(3));
	}

	// Propagate to Lower-Right cell
	if (isNotRightBoundary && isNotLowerBoundary) {
		of5678[nPos.get_value(4)].set_value(0, currentCellDistributions.f5678.get_value(0));
	}

	// Propogate to Lower-Left cell
	if (isNotLowerBoundary && isNotLeftBoundary) {
		of5678[nPos.get_value(5)].set_value(1, currentCellDistributions.f5678.get_value(1));
	}

	// Propagate to Upper-Left cell
	if (isNotLeftBoundary && isNotUpperBoundary) {
		of5678[nPos.get_value(6)].set_value(2, currentCellDistributions.f5678.get_value(2));
	}

	// Propagate to Upper-Right cell
	if (isNotUpperBoundary && isNotRightBoundary) {
		of5678[nPos.get_value(7)].set_value(3, currentCellDistributions.f5678.get_value(3));
	}

};