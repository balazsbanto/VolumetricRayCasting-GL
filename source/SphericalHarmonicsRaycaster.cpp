#include <SphericalHarmonicsRaycaster.hpp>
//#define RUN_ON_CPU
#include <Common.hpp>
namespace kernels { struct Raycaster_SphericalHarmonics; }
using namespace cl::sycl;

struct SphericalCoordinates {
	float r = 0;
	float theta = 0;
	float phi = 0;

};


const auto Y_l1_m0 = [](const SphericalCoordinates& sphericalCoordinates) {
	return 1.0f / 2.0f * float(cl::sycl::sqrt(3.0f / M_PI)) * float(cl::sycl::cos(sphericalCoordinates.theta));
};

const auto Y_l2_m0 = [](const SphericalCoordinates& sphericalCoordinates) {
	return 1.0f / 4.0f * float(cl::sycl::sqrt(5.0f / M_PI)) *
		(3.f * cl::sycl::pow(float(cl::sycl::cos(sphericalCoordinates.theta)), 2.f) - 1);
};

const auto Y_l3_m0 = [](const SphericalCoordinates& sphericalCoordinates) {
	return 1.0f / 4.0f * float(cl::sycl::sqrt(7.0f / M_PI)) *
		(5.f * cl::sycl::pow(float(cl::sycl::cos(sphericalCoordinates.theta)), 3.f)
			- 3.f * float(cl::sycl::cos(sphericalCoordinates.theta)));
};

const auto densityFunc = [](const SphericalCoordinates& sphericalCoordinates)
{
	float val = Y_l3_m0(sphericalCoordinates);
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

#ifdef RUN_ON_CPU
const auto raymarch = [](const float3& camPos, const float3& rayDirection, const float startT, const float endT,
	const float stepSize, const std::array<std::array<float, 2>, 3 >& extent, std::ofstream& rayPointsFile)
#else
const auto raymarch = [](const float3& camPos, const float3& rayDirection, const float startT, const float endT,
	const float stepSize, const std::array<std::array<float, 2>, 3 >& extent)
#endif 

{
	int saturationThreshold = 0;

	float4 finalColor(0.0f, 0.0f, 0.0f, 0.0f);
	float3 location(0.0f, 0.0f, 0.0f);

	location = camPos + startT * rayDirection;

	float current_t = startT;

	bool isSaturated = false;

	while (current_t < endT && !isSaturated)
	{
		location += stepSize * rayDirection;
		current_t += stepSize;

		if (isInside(extent, location)) {

			float4 color = colorFunc(densityFunc(transformWorldCoordinates(location)));

			finalColor += color;

			isSaturated = finalColor.r() > saturationThreshold || finalColor.g() > saturationThreshold || finalColor.b() > saturationThreshold;

		}
	}

	// normalizer according to the highest rgb value
	auto normalizer = std::max(1.0f, std::max(std::max(finalColor.r(), finalColor.g()), finalColor.b()));
	finalColor /= normalizer;
	finalColor.set_value(W, 1.f);

	return finalColor;
};

SphericalHarmonicsRaycaster::SphericalHarmonicsRaycaster(std::size_t plat,
	std::size_t dev,
	cl_bitfield type,
	QWindow* parent)
	: Raycaster(plat, dev, type, parent)
{
}

#ifdef RUN_ON_CPU
void SphericalHarmonicsRaycaster::updateSceneImpl() {

	int screen_width = width();
	int screen_height = height();
	const float aspectRatio = (float)screen_width / screen_height;
	auto ViewToWorldMtx = m_viewToWorldMtx;
	auto camPosGlm = m_vecEye;

	std::ofstream rayPointsFile("rayPointsFile.txt");

	for (int y = 0; y < screen_height; y++) {
		for (int x = 0; x < screen_width; x++) {

			int2 i{ x, y };

			glm::vec4 rayVec((2 * (i.get_value(0) + 0.5f) / (float)screen_width - 1) * aspectRatio /* * scaleFOV */,
				(1 - 2 * (i.get_value(1) + 0.5f) / (float)screen_height) /* * scaleFOV*/,
				-1.0f, 1.0f);

			// Quick switch to glm vectors to perform 4x4 matrix x vector multiplication, since SYCL has not have yet these operation
			glm::vec3 transformedCamRayDirGlm = glm::vec3(ViewToWorldMtx * rayVec) - camPosGlm;
			float3 normalizedCamRayDir = cl::sycl::normalize(float3{ transformedCamRayDirGlm.x, transformedCamRayDirGlm.y, transformedCamRayDirGlm.z });

			auto cameraPos = float3(camPosGlm.x, camPosGlm.y, camPosGlm.z);
			auto spherIntersection = getIntersections(cameraPos, normalizedCamRayDir, sphereBoundigBox);

			float4 pixelColor;
			if (spherIntersection.isIntersected && spherIntersection.t0 > 0.0 && spherIntersection.t1 > 0.0)
			{
				rayPointsFile << "Start ray: " << x << " " << y << "\n";
				pixelColor = raymarch(cameraPos, normalizedCamRayDir, spherIntersection.t0, spherIntersection.t1, stepSize, extent, rayPointsFile);
			}
			// if we are inside the spehere, we trace from the the ray's original position
			else if (spherIntersection.isIntersected && spherIntersection.t1 > 0.f)
			{
				rayPointsFile << "Start ray: " << x << " " << y << "\n";
				pixelColor = raymarch(cameraPos, normalizedCamRayDir, 0.0, spherIntersection.t1, stepSize, extent, rayPointsFile);
			}
			else
			{
				pixelColor = float4(0.f, 0.f, 0.f, 1.f);
			}

		}
	}
	rayPointsFile.close();
}
#else
void SphericalHarmonicsRaycaster::updateSceneImpl() {

	int screen_width = width();
	int screen_height = height();
	const float aspectRatio = (float)screen_width / screen_height;

	compute_queue.submit([&](cl::sycl::handler& cgh)
	{
		auto new_lattice = latticeImages[Buffer::Back]->get_access<float4, access::mode::write>(cgh);
		//float scaleFOV = tan(120.f / 2 * M_PI / 180);
		// scaleFOV?
		cgh.parallel_for<kernels::Raycaster_SphericalHarmonics>(range<2>{ new_lattice.get_range() },
			[=, ViewToWorldMtx = m_viewToWorldMtx, camPosGlm = m_vecEye, sphereBoundigBox = sphereBoundigBox, stepSize = stepSize,
			raymarch = raymarch, getIntersections = getIntersections, extent = extent
			](const item<2> i)
		{
			auto setPixelForNewLattice = [=](float4 in) { new_lattice.write((int2)i.get_id(), in); };

			glm::vec4 rayVec((2 * (i[0] + 0.5f) / (float)screen_width - 1) * aspectRatio /* * scaleFOV */,
				(1 - 2 * (i[1] + 0.5f) / (float)screen_height) /* * scaleFOV*/,
				-1.0f, 1.0f);

			// Quick switch to glm vectors to perform 4x4 matrix x vector multiplication, since SYCL has not have yet these operation
			glm::vec3 transformedCamRayDirGlm = glm::vec3(ViewToWorldMtx * rayVec) - camPosGlm;
			float3 normalizedCamRayDir = cl::sycl::normalize(float3{ transformedCamRayDirGlm.x, transformedCamRayDirGlm.y, transformedCamRayDirGlm.z });

			auto cameraPos = float3(camPosGlm.x, camPosGlm.y, camPosGlm.z);
			auto spherIntersection = getIntersections(cameraPos, normalizedCamRayDir, sphereBoundigBox);

			float4 pixelColor;
			if (spherIntersection.isIntersected && spherIntersection.t0 > 0.0 && spherIntersection.t1 > 0.0)
			{
				pixelColor = raymarch(cameraPos, normalizedCamRayDir, spherIntersection.t0, spherIntersection.t1, stepSize, extent);
			}
			// if we are inside the spehere, we trace from the the ray's original position
			else if (spherIntersection.isIntersected && spherIntersection.t1 > 0.f)
			{
				pixelColor = raymarch(cameraPos, normalizedCamRayDir, 0.0, spherIntersection.t1, stepSize, extent);
			}
			else
			{
				pixelColor = float4(0.f, 0.f, 0.f, 1.f);
			}

			// seting rgb value for every pixel
			setPixelForNewLattice(pixelColor);
		});
	});
}
#endif