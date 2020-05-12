#include <RaycasterLatticeBoltzmann2D.hpp>
#include <Lbm2D.hpp>

using namespace cl::sycl;

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

#ifdef RUN_ON_CPU
			//if (cl::sycl::fabs(location.get_value(Z)) < stepSize) {
			auto transformedToLbm = transformWorldCoordinates(location);
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
};

RaycasterLatticeBoltzmann2D::RaycasterLatticeBoltzmann2D(std::size_t plat,
    std::size_t dev,
    cl_bitfield type,
    QWindow* parent)
    : Raycaster(plat, dev, type, parent)
{
}

#ifdef RUN_ON_CPU
void RaycasterLatticeBoltzmann2D::updateSceneImpl() {

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
void RaycasterLatticeBoltzmann2D::updateSceneImpl() {

	int screen_width = width();
	int screen_height = height();
	const float aspectRatio = (float)screen_width / screen_height;

	compute_queue.submit([&](cl::sycl::handler& cgh)
	{
		auto new_lattice = latticeImages[Buffer::Back]->get_access<float4, access::mode::write>(cgh);
		//float scaleFOV = tan(120.f / 2 * M_PI / 180);
		// scaleFOV?
		cgh.parallel_for<kernels::Raycaster_Kernel>(range<2>{ new_lattice.get_range() },
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