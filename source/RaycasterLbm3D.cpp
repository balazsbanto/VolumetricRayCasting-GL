#include <RaycasterLbm3D.hpp>
#include <iomanip>
#include <Common.hpp>

namespace kernels { struct Raycaster_LBM3D; }
using namespace cl::sycl;

const auto sumF2 = [](const cl::sycl::float2& f) {
	int sum = 0;
	for (int i = 0; i < f.get_count(); i++) {
		sum += f.get_value(i);
	}
	return sum;
};

const auto sumF4 = [](const cl::sycl::float4& f) {
	int sum = 0;
	for (int i = 0; i < f.get_count(); i++) {
		sum += f.get_value(i);
	}
	return sum;
};

const auto sumF8 = [](const cl::sycl::float8& f) {
	int sum = 0;
	for (int i = 0; i < f.get_count(); i++) {
		sum += f.get_value(i);
	}
	return sum;
};

const auto getIndex = [](const cl::sycl::int3 id, const cl::sycl::int3& meshDim) {
	return id.get_value(Z) + meshDim.get_value(Z) * (id.get_value(X) + meshDim.get_value(X) * id.get_value(Y));
};

struct Distributions {
	float f0;
	cl::sycl::float4 f1to4;
	cl::sycl::float2 f56;
	cl::sycl::float8 f7to14;
	cl::sycl::float4 f15to18;
};

struct CellData
{
	Distributions distributions;
	cl::sycl::float3 velocity;
	bool cellType;

};

template <cl::sycl::access::target Target, cl::sycl::access::mode Mode>
struct DistributionBuffers {
	const cl::sycl::accessor<float, 1, Mode, Target, cl::sycl::access::placeholder::false_t>& f0;
	const cl::sycl::accessor<cl::sycl::float4, 1, Mode, Target, cl::sycl::access::placeholder::false_t>& f1to4;
	const cl::sycl::accessor<cl::sycl::float2, 1, Mode, Target, cl::sycl::access::placeholder::false_t>& f56;
	const cl::sycl::accessor<cl::sycl::float8, 1, Mode, Target, cl::sycl::access::placeholder::false_t>& f7to14;
	const cl::sycl::accessor<cl::sycl::float4, 1, Mode, Target, cl::sycl::access::placeholder::false_t>& f15to18;

};

const auto sumDistributions = [](const Distributions& distributions) {
	return distributions.f0 + sumF4(distributions.f1to4) + sumF2(distributions.f56) + sumF8(distributions.f7to14) + sumF4(distributions.f15to18);
};

const auto colorFunc = [](cl::sycl::float3 inVelocity, bool isBoundary) {
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


const auto computefEq = [](const float rho, const float weight, const cl::sycl::float3 dir, const cl::sycl::float3 velocity) {

	float u2 = cl::sycl::dot(velocity, velocity);
	float eu = cl::sycl::dot(dir, velocity);
	return rho * weight * (1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * u2);
};

// Unit direction vectors and their buffers
// Weights
const float weight_0 = 1.f / 3.f;
const float weight_1to6 = 1.f / 18.f;
const float weight_7to18 = 1.f / 36.f;

const float rho = 10.f;

const float f0_EQ = rho * weight_0;
const float f1to6_EQ = rho * weight_1to6;
const float f7to18_EQ = rho * weight_7to18;

const std::array<float, 19> h_weight{ weight_0, weight_1to6, weight_1to6, weight_1to6, weight_1to6, weight_1to6, weight_1to6,
							weight_7to18, weight_7to18, weight_7to18, weight_7to18, weight_7to18, weight_7to18,
							weight_7to18, weight_7to18, weight_7to18, weight_7to18, weight_7to18, weight_7to18, };

const std::array<cl::sycl::float3, 19> h_dir{ cl::sycl::float3{0, 0, 0}, cl::sycl::float3{1, 0, 0}, cl::sycl::float3{-1, 0, 0}, cl::sycl::float3{0, 1, 0}, cl::sycl::float3{0, -1, 0}, cl::sycl::float3{0, 0, 1}, cl::sycl::float3{0, 0, -1},cl::sycl::float3{1, 1, 0},
		cl::sycl::float3{-1, -1, 0}, cl::sycl::float3{1, -1, 0}, cl::sycl::float3{-1, 1, 0}, cl::sycl::float3{1, 0, 1}, cl::sycl::float3{-1, 0, -1}, cl::sycl::float3{1, 0, -1}, cl::sycl::float3{-1, 0, 1}, cl::sycl::float3{0, 1, 1}, cl::sycl::float3{0, -1, -1}, cl::sycl::float3{0, 1, -1},
		cl::sycl::float3{0, -1, 1}, };

const auto calculateVelocity = [](const Distributions& cellDistributions, const std::array<cl::sycl::float3, 19> unitDirVectors) {
	using namespace cl::sycl;
	float3 velocity{ 0,0,0, };

	velocity = cellDistributions.f0 * unitDirVectors[0];
	int vIndex = 0;
	for (int i = 0; i < cellDistributions.f1to4.get_count(); i++) {
		velocity += cellDistributions.f1to4.get_value(i) * unitDirVectors[vIndex++];
	}
	
	for (int i = 0; i < cellDistributions.f56.get_count(); i++) {
		velocity += cellDistributions.f56.get_value(i) * unitDirVectors[vIndex++];
	}

	for (int i = 0; i < cellDistributions.f7to14.get_count(); i++) {
		velocity += cellDistributions.f7to14.get_value(i) * unitDirVectors[vIndex++];
	}

	for (int i = 0; i < cellDistributions.f15to18.get_count(); i++) {
		velocity += cellDistributions.f15to18.get_value(i) * unitDirVectors[vIndex++];
	}

	return velocity;
};

const auto collide = [h_dir = h_dir, h_weight = h_weight](const Distributions& inCellDistributions, const bool cellType) {
	using namespace cl::sycl;

	// (1/relaxation time) Related to viscosity 
	float omega = 1.2f;

	float rho;
	float3 u;

	Distributions outDistribution;

	//boundary
	if (cellType) {
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
		u = float3{ 0.f, 0.f, 0.f };
		outDistribution = inCellDistributions;
	}
	// fluid
	else
	{
		rho = sumDistributions(inCellDistributions);
		u = calculateVelocity(inCellDistributions, h_dir) / rho;
		float fEq0;
		float4 fEq1to4;	
		float2 fEq56;
		float8 fEq7to14;
		float4 fEq15to18;

		fEq0 = computefEq(rho, h_weight[0], h_dir[0], u);

		fEq1to4.set_value(0, computefEq(rho, h_weight[1], h_dir[1], u));
		fEq1to4.set_value(1, computefEq(rho, h_weight[2], h_dir[2], u));
		fEq1to4.set_value(2, computefEq(rho, h_weight[3], h_dir[3], u));
		fEq1to4.set_value(3, computefEq(rho, h_weight[4], h_dir[4], u));

		fEq56.set_value(0, computefEq(rho, h_weight[5], h_dir[5], u));
		fEq56.set_value(1, computefEq(rho, h_weight[6], h_dir[6], u));

		fEq7to14.set_value(0, computefEq(rho, h_weight[7], h_dir[7], u));
		fEq7to14.set_value(1, computefEq(rho, h_weight[8], h_dir[8], u));
		fEq7to14.set_value(2, computefEq(rho, h_weight[9], h_dir[9], u));
		fEq7to14.set_value(3, computefEq(rho, h_weight[10], h_dir[10], u));
		fEq7to14.set_value(4, computefEq(rho, h_weight[11], h_dir[11], u));
		fEq7to14.set_value(5, computefEq(rho, h_weight[12], h_dir[12], u));
		fEq7to14.set_value(6, computefEq(rho, h_weight[13], h_dir[13], u));
		fEq7to14.set_value(7, computefEq(rho, h_weight[14], h_dir[14], u));

		fEq15to18.set_value(0, computefEq(rho, h_weight[15], h_dir[15], u));
		fEq15to18.set_value(1, computefEq(rho, h_weight[16], h_dir[16], u));
		fEq15to18.set_value(2, computefEq(rho, h_weight[17], h_dir[17], u));
		fEq15to18.set_value(3, computefEq(rho, h_weight[18], h_dir[18], u));

		outDistribution = Distributions{ (1 - omega) * inCellDistributions.f0 + omega * fEq0,
										(1 - omega) * inCellDistributions.f1to4 + omega * fEq1to4,
										(1 - omega) * inCellDistributions.f56 + omega * fEq56,
										(1 - omega) * inCellDistributions.f7to14 + omega * fEq7to14,
										(1 - omega) * inCellDistributions.f15to18 + omega * fEq15to18, };
	}

	return CellData{ outDistribution, u, cellType };
};

template <cl::sycl::access::target Target>
struct streamToNeighbours {
	void operator()(const cl::sycl::int3 id,  const cl::sycl::int3 &meshDim,
		const Distributions& currentCellDistributions,
		const DistributionBuffers<Target, cl::sycl::access::mode::discard_write>& outDistributionBuffers) const {

		using namespace cl::sycl;
		// TODO: Streaming near the border could be wrong.
		if (id.get_value(X) == 0 || id.get_value(X) == (meshDim.get_value(X) - 1)
			|| id.get_value(Y) == 0 || id.get_value(Y) == (meshDim.get_value(Y) - 1)
			|| id.get_value(Z) == 0 || id.get_value(Z) == (meshDim.get_value(Z) - 1)) {
			return;
		}

		for (int i = 0; i < h_dir.size(); i++) {
			int3 dir3 = id + int3{ h_dir[i].get_value(X), h_dir[i].get_value(Y), h_dir[i].get_value(Z) };
			int pos = getIndex(dir3, meshDim);
			switch (i)
			{
			case 0:
				outDistributionBuffers.f0[pos] = currentCellDistributions.f0;
				break;
			case 1:
				outDistributionBuffers.f1to4[pos].set_value(0, currentCellDistributions.f1to4.get_value(0));
				break;
			case 2:
				outDistributionBuffers.f1to4[pos].set_value(1, currentCellDistributions.f1to4.get_value(1));
				break;
			case 3:
				outDistributionBuffers.f1to4[pos].set_value(2, currentCellDistributions.f1to4.get_value(2));
				break;
			case 4:
				outDistributionBuffers.f1to4[pos].set_value(3, currentCellDistributions.f1to4.get_value(3));
				break;
			case 5:
				outDistributionBuffers.f56[pos].set_value(0, currentCellDistributions.f56.get_value(0));
				break;
			case 6:
				outDistributionBuffers.f56[pos].set_value(1, currentCellDistributions.f56.get_value(1));
				break;
			case 7:
				outDistributionBuffers.f7to14[pos].set_value(0, currentCellDistributions.f7to14.get_value(0));
				break;
			case 8:
				outDistributionBuffers.f7to14[pos].set_value(1, currentCellDistributions.f7to14.get_value(1));
				break;
			case 9:
				outDistributionBuffers.f7to14[pos].set_value(2, currentCellDistributions.f7to14.get_value(2));
				break;
			case 10:
				outDistributionBuffers.f7to14[pos].set_value(3, currentCellDistributions.f7to14.get_value(3));
				break;
			case 11:
				outDistributionBuffers.f7to14[pos].set_value(4, currentCellDistributions.f7to14.get_value(4));
				break;
			case 12:
				outDistributionBuffers.f7to14[pos].set_value(5, currentCellDistributions.f7to14.get_value(5));
				break;
			case 13:
				outDistributionBuffers.f7to14[pos].set_value(6, currentCellDistributions.f7to14.get_value(6));
				break;
			case 14:
				outDistributionBuffers.f7to14[pos].set_value(7, currentCellDistributions.f7to14.get_value(7));
				break;
			case 15:
				outDistributionBuffers.f15to18[pos].set_value(0, currentCellDistributions.f15to18.get_value(0));
				break;
			case 16:
				outDistributionBuffers.f15to18[pos].set_value(1, currentCellDistributions.f15to18.get_value(1));
				break;
			case 17:
				outDistributionBuffers.f15to18[pos].set_value(2, currentCellDistributions.f15to18.get_value(2));
				break;
			case 18:
				outDistributionBuffers.f15to18[pos].set_value(3, currentCellDistributions.f15to18.get_value(3));
				break;
			}
		}
	}
};

const auto transformWorldCoordinates = [](const float3& worldLocation, const int extentLim, const int lbmSize) {
	return float3{ worldLocation.get_value(X) + extentLim, extentLim - worldLocation.get_value(Y), worldLocation.get_value(Z) + extentLim }
	*(lbmSize / (2 * extentLim));
};

template <cl::sycl::access::target Target>
struct Lbm2DSpaceAccessors {
	const DistributionBuffers<Target, cl::sycl::access::mode::discard_write>& distributions;
	const cl::sycl::accessor<cl::sycl::float3, 1, cl::sycl::access::mode::discard_write, Target,
		cl::sycl::access::placeholder::false_t>& velocity;
	const cl::sycl::accessor<bool, 1, cl::sycl::access::mode::read, Target,
		cl::sycl::access::placeholder::false_t>& cellType;


};

template <cl::sycl::access::target Target>
struct raymarch {
	float4 operator()(const float3& camPos, const float3& rayDirection, const float startT, const float endT,
		const float stepSize, const std::array<std::array<float, 2>, 3 >& extent, const cl::sycl::int3 &meshDim,
		const DistributionBuffers<Target, access::mode::read>& inDistributionBuffers,
		const Lbm2DSpaceAccessors<Target>& spaceAccessors
#ifdef RUN_ON_CPU
		, std::ofstream& rayPointsFile
#endif // RUN_ON_CPU
		) const
	{
		int saturationThreshold = 0;

		float4 finalColor(0.0f, 0.0f, 0.0f, 0.0f);
		float3 location(0.0f, 0.0f, 0.0f);

		location = camPos + startT * rayDirection;

		float current_t = startT;

		bool isSaturated = false;

		while (current_t < endT /*&& !isSaturated*/)
		{
			location += stepSize * rayDirection;
			current_t += stepSize;

			if (isInside(extent, location)) {

				//float4 color(0.0f, 0.0f, 0.0f, 0.0f);
				auto lbmSpaceCoordinates = transformWorldCoordinates(location, int(extent[0][1]), meshDim.get_value(0));
				int3 id{ int(lbmSpaceCoordinates.get_value(X)), int(lbmSpaceCoordinates.get_value(Y)),  int(lbmSpaceCoordinates.get_value(Y)) };

				int pos = getIndex(id, meshDim);

				auto cellAfterCollision = collide(Distributions{ inDistributionBuffers.f0[pos], inDistributionBuffers.f1to4[pos],
					inDistributionBuffers.f56[pos], inDistributionBuffers.f7to14[pos], inDistributionBuffers.f15to18[pos] }, spaceAccessors.cellType[pos]);

				streamToNeighbours<Target >()(id, meshDim, cellAfterCollision.distributions, spaceAccessors.distributions);

				spaceAccessors.velocity[pos] = cellAfterCollision.velocity;

				float4 color = colorFunc(cellAfterCollision.velocity, cellAfterCollision.cellType);
				//float4 color = colorFunc(densityFunc(transformWorldCoordinates(location)));

				finalColor += color;

#ifdef RUN_ON_CPU
				//if (cl::sycl::fabs(location.get_value(Z)) < stepSize) {
				auto transformedToLbm = transformWorldCoordinates(location, int(extent[0][1]), meshDim.get_value(0));
				rayPointsFile << (int)transformedToLbm.get_value(X) << " " << (int)transformedToLbm.get_value(Y) << " " << transformedToLbm.get_value(Z) << "\n";
				//}
#endif
			}
		}

		// normalizer according to the highest rgb value
		auto normalizer = std::max(1.0f, std::max(std::max(finalColor.r(), finalColor.g()), finalColor.b()));
		finalColor /= normalizer;
		finalColor.set_value(W, 1.f);

		return finalColor;
	}
};

RaycasterLbm3D::RaycasterLbm3D(std::size_t plat,
	std::size_t dev,
	cl_bitfield type,
	QWindow* parent)
	: Raycaster(plat, dev, type, parent)
{
}

#ifdef RUN_ON_CPU
void RaycasterLbm3D::updateSceneImpl() {

	auto if0 = f0_buffers[Buffer::Front]->get_access<access::mode::read>();
	auto if1234 = f1234_buffers[Buffer::Front]->get_access<access::mode::read>();
	auto if5678 = f5678_buffers[Buffer::Front]->get_access<access::mode::read>();
	auto type = type_buffer.get_access<access::mode::read>();

	// Output
	auto of0 = f0_buffers[Buffer::Back]->get_access<access::mode::discard_write>();
	auto of1234 = f1234_buffers[Buffer::Back]->get_access<access::mode::discard_write>();
	auto of5678 = f5678_buffers[Buffer::Back]->get_access<access::mode::discard_write>();
	auto velocity_out = velocity_buffer->get_access<access::mode::discard_write>();


	int screen_width = width();
	int screen_height = height();
	const float aspectRatio = (float)screen_width / screen_height;
	auto ViewToWorldMtx = m_viewToWorldMtx;
	auto camPosGlm = m_vecEye;

	std::ofstream rayPointsFile("rayPointsFile.txt");

	for (int y = 0; y < screen_height; y++) {
		for (int x = 0; x < screen_width; x++) {

			int2 i{ x, y };

			glm::vec4 rayVec((2 * (i.get_value(0) + 0.5f) / (float)screenSize.width - 1) * aspectRatio /* * scaleFOV */,
				(1 - 2 * (i.get_value(1) + 0.5f) / (float)screenSize.height) /* * scaleFOV*/,
				-1.0f, 1.0f);

			// Quick switch to glm vectors to perform 4x4 matrix x vector multiplication, since SYCL has not have yet these operation
			glm::vec3 transformedCamRayDirGlm = glm::vec3(ViewToWorldMtx * rayVec) - camPosGlm;
			float3 normalizedCamRayDir = cl::sycl::normalize(float3{ transformedCamRayDirGlm.x, transformedCamRayDirGlm.y, transformedCamRayDirGlm.z });

			auto cameraPos = float3(camPosGlm.x, camPosGlm.y, camPosGlm.z);
			auto spherIntersection = getIntersections(cameraPos, normalizedCamRayDir, sphereBoundigBox);

			float4 pixelColor;
			if (spherIntersection.isIntersected)
			{
				if (spherIntersection.t0 < 0.f) {
					spherIntersection.t0 = 0.f;
				}

				pixelColor = raymarch<access::target::host_buffer>()
					(cameraPos, normalizedCamRayDir, spherIntersection.t0, spherIntersection.t1, stepSize, extent, screenSize,
						DistributionBuffers<access::target::host_buffer, access::mode::read>{ if0, if1234, if5678 },
						Lbm2DSpaceAccessors< access::target::host_buffer > {
					DistributionBuffers<access::target::host_buffer, access::mode::discard_write>{ of0, of1234, of5678 },
						velocity_out, type },
						rayPointsFile
						);
			}
			// if we are inside the spehere, we trace from the the ray's original position
			else
			{
				pixelColor = float4(0.f, 0.f, 0.f, 1.f);
			}

		}
	}
	rayPointsFile.close();
}
#else
void RaycasterLbm3D::updateSceneImpl() {

	const float aspectRatio = screenSize.aspectRatio();

	compute_queue.submit([&](cl::sycl::handler& cgh)
	{
		// Input buffers
		auto if0 = f0_buffers[Buffer::Front]->get_access<access::mode::read>(cgh);
		auto if1to4 = f1to4_buffers[Buffer::Front]->get_access<access::mode::read>(cgh);
		auto if56 = f56_buffers[Buffer::Front]->get_access<access::mode::read>(cgh);
		auto if7to14 = f7to14_buffers[Buffer::Front]->get_access<access::mode::read>(cgh);
		auto if15to18 = f15to18_buffers[Buffer::Front]->get_access<access::mode::read>(cgh);

		auto cellType = type_buffer.get_access<access::mode::read>(cgh);

		// Output buffers
		auto of0 = f0_buffers[Buffer::Back]->get_access<access::mode::discard_write>(cgh);
		auto of1to4 = f1to4_buffers[Buffer::Back]->get_access<access::mode::discard_write>(cgh);
		auto of56 = f56_buffers[Buffer::Back]->get_access<access::mode::discard_write>(cgh);
		auto of7to14 = f7to14_buffers[Buffer::Back]->get_access<access::mode::discard_write>(cgh);
		auto of15to18 = f15to18_buffers[Buffer::Back]->get_access<access::mode::discard_write>(cgh);

		auto velocity_out =	velocity_buffer->get_access<access::mode::discard_write>(cgh);
		

		auto new_lattice = latticeImages[Buffer::Back]->get_access<float4, access::mode::discard_write>(cgh);


		cgh.parallel_for<kernels::Raycaster_LBM3D>(range<2>{ new_lattice.get_range() },
			[=, ViewToWorldMtx = m_viewToWorldMtx, camPosGlm = m_vecEye, sphereBoundigBox = sphereBoundigBox, stepSize = stepSize,
			extent = extent, screenSize = screenSize, meshDim = meshDim](const item<2> i)
		{

			glm::vec4 rayVec((2 * (i[0] + 0.5f) / (float)screenSize.width - 1) * aspectRatio /* * scaleFOV */,
				(1 - 2 * (i[1] + 0.5f) / (float)screenSize.height) /* * scaleFOV*/,
				-1.0f, 1.0f);

			// Quick switch to glm vectors to perform 4x4 matrix x vector multiplication, since SYCL has not have yet these operation
			glm::vec3 transformedCamRayDirGlm = glm::vec3(ViewToWorldMtx * rayVec) - camPosGlm;
			float3 normalizedCamRayDir = cl::sycl::normalize(float3{ transformedCamRayDirGlm.x, transformedCamRayDirGlm.y, transformedCamRayDirGlm.z });

			auto cameraPos = float3(camPosGlm.x, camPosGlm.y, camPosGlm.z);
			auto spherIntersection = getIntersections(cameraPos, normalizedCamRayDir, sphereBoundigBox);

			float4 pixelColor;
			if (spherIntersection.isIntersected)
			{
				// if we are already inside the bounding sphere, we start from the ray current position
				if (spherIntersection.t0 < 0.f) {
					spherIntersection.t0 = 0.f;
				}

				pixelColor = raymarch<access::target::global_buffer>()
					(cameraPos, normalizedCamRayDir, spherIntersection.t0, spherIntersection.t1, stepSize, extent, meshDim,
						DistributionBuffers<access::target::global_buffer, access::mode::read> { if0, if1to4, if56, if7to14, if15to18},
						Lbm2DSpaceAccessors< access::target::global_buffer > {
							DistributionBuffers<access::target::global_buffer, access::mode::discard_write>{ of0, of1to4, of56, of7to14, of15to18},
							velocity_out, cellType}	);
			}
			else
			{
				pixelColor = float4(0.f, 0.f, 0.f, 1.f);
			}

			auto setPixelForNewLattice = [=](float4 in) { new_lattice.write((int2)i.get_id(), in); };
			// seting rgb value for every pixel
			setPixelForNewLattice(pixelColor);
		});
	});
}
#endif


void RaycasterLbm3D::resetScene() {
	
	using namespace cl::sycl;
	meshDim = int3{ screenSize.width, screenSize.width, screenSize.width };
	size_t meshSize = meshDim.get_value(X) * meshDim.get_value(Y) * meshDim.get_value(Z);
	// Initial velocity is 0
	type_host = new bool[meshSize];
	f0_host[Buffer::Front] = std::vector<float>(meshSize, f0_EQ);
	f1to4_host[Buffer::Front] = std::vector<float4>(meshSize, float4{ f1to6_EQ });
	f56_host[Buffer::Front] = std::vector<float2>(meshSize, float2{ f1to6_EQ });
	f7to14_host[Buffer::Front] = std::vector<float8>(meshSize, float8{ f7to18_EQ });
	f15to18_host[Buffer::Front] = std::vector<float4>(meshSize, float4{ f7to18_EQ });

	f0_host[Buffer::Back] = f0_host[Buffer::Front];
	f1to4_host[Buffer::Back] = f1to4_host[Buffer::Front];
	f56_host[Buffer::Back] = f56_host[Buffer::Front];
	f56_host[Buffer::Back] = f56_host[Buffer::Front];
	f15to18_host[Buffer::Back] = f15to18_host[Buffer::Front];

	f0_buffers[Buffer::Front] = std::make_unique<buffer<float, 1>>(f0_host[Buffer::Front].data(), range<1> {meshSize});
	f1to4_buffers[Buffer::Front] = std::make_unique<buffer<float4, 1>>(f1to4_host[Buffer::Front].data(), range<1> {meshSize});
	f56_buffers[Buffer::Front] = std::make_unique<buffer<float2, 1>>(f56_host[Buffer::Front].data(), range<1> { meshSize});
	f7to14_buffers[Buffer::Front] = std::make_unique<buffer<float8, 1>>(f7to14_host[Buffer::Front].data(), range<1> { meshSize});
	f15to18_buffers[Buffer::Front] = std::make_unique<buffer<float4, 1>>(f15to18_host[Buffer::Front].data(), range<1> { meshSize});


	f0_buffers[Buffer::Back] = std::make_unique<buffer<float, 1>>(f0_host[Buffer::Back].data(), range<1> {meshSize});
	f1to4_buffers[Buffer::Back] = std::make_unique<buffer<float4, 1>>(f1to4_host[Buffer::Back].data(), range<1> {meshSize});
	f56_buffers[Buffer::Back] = std::make_unique<buffer<float2, 1>>(f56_host[Buffer::Back].data(), range<1> { meshSize});
	f7to14_buffers[Buffer::Back] = std::make_unique<buffer<float8, 1>>(f7to14_host[Buffer::Back].data(), range<1> { meshSize});
	f15to18_buffers[Buffer::Back] = std::make_unique<buffer<float4, 1>>(f15to18_host[Buffer::Back].data(), range<1> { meshSize});

	velocity_host = std::vector<float3>(meshSize, float3{ 0.f, 0.f, 0.f });
	velocity_buffer = std::make_unique< buffer<float3, 1>>(velocity_host.data(), range<1> { meshSize});



	for (int y = 0; y < meshDim.get_value(Y); y++) {
		for (int x = 0; x < meshDim.get_value(X); x++) {
			for (int z = 0; z < meshDim.get_value(Z); z++) {

				int pos = getIndex(int3{ x, y, y }, meshDim);

				// Initialize boundary cells
				if (x == 0 || x == (meshDim.get_value(X) - 1) || y == 0 || y == (meshDim.get_value(Y) - 1)
					|| z == 0 || z == (meshDim.get_value(Z) - 1))
				{
					type_host[pos] = true;
				}

				// Initialize fluid cells
				else
				{
					type_host[pos] = false;
				}
			}
		}
	}

	type_buffer = buffer<bool, 1>{ type_host, range<1> {meshSize} };

	writeOutputsToFile();
	setInput();
	writeOutputsToFile();
}

void RaycasterLbm3D::setInput() {

	using namespace cl::sycl;
	int x = meshDim.get_value(X) / 2;
	int y = meshDim.get_value(Y) - 1 - meshDim.get_value(Y) / 2;
	int z = meshDim.get_value(Z) /2;

	int pos = getIndex(int3{ x, y, z }, meshDim);

	auto if0 = f0_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read_write>();
	auto if1to4 = f1to4_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read_write>();
	auto if56 = f56_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read_write>();
	auto if7to14 = f7to14_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read_write>();
	auto if15to18 = f15to18_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read_write>();

	auto velocity_out = velocity_buffer->get_access<access::mode::read>();

	// Calculate density from input distribution
	float rho = sumDistributions(Distributions{ if0[pos], if1to4[pos], if56[pos], if7to14[pos], if15to18[pos]});

	// Increase the speed by input speed
	//velocity_out[pos] += dragVelocity;

	float3 newVel = velocity_out[pos] + float3{ 1.f, 1.f, 1.f };

	// Calculate new distribution based on input spee

	if0[pos] = computefEq(rho, h_weight[0], h_dir[0], newVel);

	if1to4[pos].set_value(0, computefEq(rho, h_weight[1], h_dir[1], newVel));
	if1to4[pos].set_value(1, computefEq(rho, h_weight[2], h_dir[2], newVel));
	if1to4[pos].set_value(2, computefEq(rho, h_weight[3], h_dir[3], newVel));
	if1to4[pos].set_value(3, computefEq(rho, h_weight[4], h_dir[4], newVel));

	if56[pos].set_value(0, computefEq(rho, h_weight[5], h_dir[5], newVel));
	if56[pos].set_value(1, computefEq(rho, h_weight[6], h_dir[6], newVel));

	if7to14[pos].set_value(0, computefEq(rho, h_weight[7], h_dir[7], newVel));
	if7to14[pos].set_value(1, computefEq(rho, h_weight[8], h_dir[8], newVel));
	if7to14[pos].set_value(2, computefEq(rho, h_weight[9], h_dir[9], newVel));
	if7to14[pos].set_value(3, computefEq(rho, h_weight[10], h_dir[10], newVel));
	if7to14[pos].set_value(4, computefEq(rho, h_weight[11], h_dir[11], newVel));
	if7to14[pos].set_value(5, computefEq(rho, h_weight[12], h_dir[12], newVel));
	if7to14[pos].set_value(6, computefEq(rho, h_weight[13], h_dir[13], newVel));
	if7to14[pos].set_value(7, computefEq(rho, h_weight[14], h_dir[14], newVel));


	if15to18[pos].set_value(0, computefEq(rho, h_weight[15], h_dir[15], newVel));
	if15to18[pos].set_value(1, computefEq(rho, h_weight[16], h_dir[16], newVel));
	if15to18[pos].set_value(2, computefEq(rho, h_weight[17], h_dir[17], newVel));
	if15to18[pos].set_value(3, computefEq(rho, h_weight[18], h_dir[18], newVel));
}

void RaycasterLbm3D::swapDataBuffers() {
	std::swap(f0_buffers[Buffer::Front], f0_buffers[Buffer::Back]);
	std::swap(f1to4_buffers[Buffer::Front], f1to4_buffers[Buffer::Back]);
	std::swap(f56_buffers[Buffer::Front], f56_buffers[Buffer::Back]);
	std::swap(f7to14_buffers[Buffer::Front], f7to14_buffers[Buffer::Back]);
	std::swap(f15to18_buffers[Buffer::Front], f15to18_buffers[Buffer::Back]);
}

void RaycasterLbm3D::writeOutputsToFile() {
#ifndef WRITE_OUTPUT_TO_FILE
	return;
#endif // !WRITE_OUTPUT_TO_FILE

	static int fileIndex = 0;

	auto f0 = f0_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read>();
	auto f1to4 = f1to4_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read>();
	auto f56 = f56_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read>();
	auto f7to14 = f7to14_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read>();
	auto f15to18 = f15to18_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read>();

	auto velocity = velocity_buffer->get_access<cl::sycl::access::mode::read>();

	std::ofstream f0_file("of0_" + std::to_string(fileIndex) + ".txt");
	std::ofstream f1to4_file("of1to4_" + std::to_string(fileIndex) + ".txt");
	std::ofstream f56_file("of56_" + std::to_string(fileIndex) + ".txt");
	std::ofstream f7to14_file("of7to14_" + std::to_string(fileIndex) + ".txt");
	std::ofstream f15to18_file("of15to18_" + std::to_string(fileIndex) + ".txt");
	std::ofstream velocity_file("velocity_" + std::to_string(fileIndex) + ".txt");


	for (int i = 0; i < f0.get_count(); i++) {
		f0_file << std::setprecision(5) << (float)f0[i] << "\n";
	}

	for (int i = 0; i < f1to4.get_count(); i++) {
		//qDebug() << f1234[i].get_value(0) << "\n";
		f1to4_file << std::setprecision(5) << f1to4[i].get_value(0) << "\t" << f1to4[i].get_value(1) << "\t" << f1to4[i].get_value(2) << "\t" << f1to4[i].get_value(3) << "\n";
	}

	for (int i = 0; i < f56.get_count(); i++) {
		f56_file << std::setprecision(5) << f56[i].get_value(0) << "\t" << f56[i].get_value(1) <<  "\n";
	}


	for (int i = 0; i < f7to14.get_count(); i++) {
		//qDebug() << f1234[i].get_value(0) << "\n";
		f7to14_file << std::setprecision(5) << f7to14[i].get_value(0) << "\t" << f7to14[i].get_value(1) << "\t" << f7to14[i].get_value(2) << "\t" << f7to14[i].get_value(3) << 
			f7to14[i].get_value(4) << "\t" << f7to14[i].get_value(5) << "\t" << f7to14[i].get_value(6) << "\t" << f7to14[i].get_value(7)  << "\n";
	}

	for (int i = 0; i < f15to18.get_count(); i++) {
		//qDebug() << f1234[i].get_value(0) << "\n";
		f15to18_file << std::setprecision(5) << f15to18[i].get_value(0) << "\t" << f15to18[i].get_value(1) << "\t" << f15to18[i].get_value(2) << "\t" << f15to18[i].get_value(3) << "\n";
	}


	for (int i = 0; i < velocity.get_count(); i++) {
		velocity_file << std::setprecision(5) << velocity[i].get_value(0) << "\t" << velocity[i].get_value(1) << "\t" << velocity[i].get_value(2) << "\n";
	}

	f0_file.close();
	f1to4_file.close();
	f56_file.close();
	f7to14_file.close();
	f15to18_file.close();
	velocity_file.close();
}