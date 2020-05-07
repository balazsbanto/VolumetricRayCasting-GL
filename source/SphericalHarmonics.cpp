#include <SphericalHarmonics.hpp>

using namespace cl::sycl;
const int X = 0;
const int Y = 1;
const int Z = 2;
const int W = 3;

const auto densityFunc = [](const float& r, const float& theta, const float& /*phi*/)
{
	float sqrt3fpi = cl::sycl::sqrt(3.0f / M_PI);
	float val = 1.0f / 2.0f * sqrt3fpi * cl::sycl::cos(theta); // Y(l = 1, m = 0)
	float result = cl::sycl::fabs(2 * cl::sycl::fabs(val) - r);

	if (result < 0.01f)	// thickness of shell 
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

//		// Start raymarch lambda
const auto m_raymarch = [](const float3& camPos, const float3& rayDirection, const float startT, const float endT, const float deltaS)
{
	int saturationThreshold = 0;

	float4 finalColor(0.0f, 0.0f, 0.0f, 0.0f);
	float3 location(0.0f, 0.0f, 0.0f);

	location = camPos + startT * rayDirection;

	float current_t = startT;

	bool isSaturated = false;

	while (current_t < endT && !isSaturated)
	{
		location += deltaS * rayDirection;
		current_t += deltaS;

		// check if it is inside
		//if (!IsOutside(location))
		/*float x = location.x();
		float y = location.y();
		float z = location.z();*/
		//if (x < extent.m_maxX)
		//if ((x < extent.m_maxX) && y < (extent.m_maxY) && (z < extent.m_maxZ) &&
		//	(x > extent.m_minX) && (y > extent.m_minY) && (z > extent.m_minZ))
		//{
			// Convert to spherical coordinated
			//float r = sqrt(location.x*location.x + location.y*location.y + location.z*location.z);

		float r = cl::sycl::length(location);
		float theta = cl::sycl::acos(location.get_value(Z) / r); // *180 / 3.1415926f; // convert to degrees?
		float phi = cl::sycl::atan2(location.get_value(Y), location.get_value(X)); // *180 / 3.1415926f;


		float4 color = colorFunc(densityFunc(r, theta, phi));


		finalColor += color;
		//} // end if check isInside

		isSaturated = finalColor.r() > saturationThreshold || finalColor.g() > saturationThreshold	|| finalColor.b() > saturationThreshold;
	}

	// normalizer according to the highest rgb value
	auto normalizer = std::max(1.0f, std::max(std::max(finalColor.r(), finalColor.g()), finalColor.b()));
	finalColor /= normalizer;
	finalColor.set_value(W, 1.f);

	return finalColor;
};

struct SphereIntersection {
	bool isIntersected = false;
	float t0 = -1E+36f;
	float t1 = -1E+36f;

};
auto getIntersections = [](const float3& rayorig, const float3& raydir, const float3& sphereCenter,
	const float sphereRadius2) {
	float3 l = sphereCenter - rayorig;
	float tca = cl::sycl::dot(l, raydir);
	float d2 = cl::sycl::dot(l, l) - tca * tca;

	SphereIntersection spherIntersection;
	if ((sphereRadius2 - d2) < 0.0001f) {
		spherIntersection.isIntersected = false;

	}
	else {
		float thc = cl::sycl::sqrt(sphereRadius2 - d2);
		spherIntersection.isIntersected = true;
		spherIntersection.t0 = tca - thc;
		spherIntersection.t1 = tca + thc;
	}

	return spherIntersection;

};


SphericalHarmonics::SphericalHarmonics(std::size_t plat,
    std::size_t dev,
    cl_bitfield type,
    QWindow* parent)
    : Raycaster(plat, dev, type, parent)
{
}

void SphericalHarmonics::mouseDragImpl(QMouseEvent* event_in) {
	phi = (event_in->x() - mousePos.x());
	theta = (event_in->y() - mousePos.y());
}

void SphericalHarmonics::updateSceneImpl() {
		compute_queue.submit([&](cl::sycl::handler& cgh)
		{
			auto new_lattice = latticeImages[Buffer::Back]->get_access<float4, access::mode::write>(cgh);

			auto aspectRatio = (float)new_lattice.get_range()[0] / new_lattice.get_range()[1];
			//float scaleFOV = tan(120.f / 2 * M_PI / 180);
			// scaleFOV?
			cgh.parallel_for<kernels::SphericalHarmonics_Kernel>(range<2>{ new_lattice.get_range() },
				[=, ViewToWorldMtx = m_viewToWorldMtx, camPosGlm = m_vecEye, sphereCenter = glm::vec3(0.f, 0.f, 0.f), sphereRadius2 = 1.96f, raymarch = m_raymarch, deltaS = 0.02f,
				getIntersections = getIntersections
				](const item<2> i)
			{
				int2 pixelIndex = i.get_id();
				auto setPixelForNewLattice = [=](float4 in) { new_lattice.write((int2)i.get_id(), in); };

				glm::vec4 rayVec((2 * (i[0] + 0.5f) / (float)new_lattice.get_range()[0] - 1)* aspectRatio /* * scaleFOV */,
					(1 - 2 * (i[1] + 0.5f) / (float)new_lattice.get_range()[1]) /* * scaleFOV*/,
					-1.0f, 1.0f);

				glm::vec3 transformedCamRayDirGlm = glm::vec3(ViewToWorldMtx * rayVec) - camPosGlm;
				float3 normalizedCamRayDir = cl::sycl::normalize(float3{ transformedCamRayDirGlm.x, transformedCamRayDirGlm.y, transformedCamRayDirGlm.z });

				auto cameraPos = float3(camPosGlm.x, camPosGlm.y, camPosGlm.z);
				auto spherIntersection = getIntersections(cameraPos, normalizedCamRayDir, float3(sphereCenter.x, sphereCenter.y, sphereCenter.z), sphereRadius2);

				float4 pixelColor;
				if (spherIntersection.isIntersected && spherIntersection.t0 > 0.0 && spherIntersection.t1 > 0.0)
				{
					pixelColor = raymarch(cameraPos, normalizedCamRayDir, spherIntersection.t0, spherIntersection.t1, deltaS);
				}
				// if we are inside the spehere, we trace from the the ray's original position
				else if (spherIntersection.isIntersected && spherIntersection.t1 > 0.f)
				{
					pixelColor = raymarch(cameraPos, normalizedCamRayDir, 0.0, spherIntersection.t1, deltaS);
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