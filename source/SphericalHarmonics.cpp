#include <SphericalHarmonics.hpp>

using namespace cl::sycl;
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
    //		// Start raymarch lambda
		auto m_raymarch = [](const cl::sycl::float3& camPos, const cl::sycl::float3& rayDirection, const float startT, const float endT, const float deltaS)
		{
			int saturationThreshold = 0;
			// example lambda functions that could be given by the user
			// density function(spherical harminics) inside the extent
			auto densityFunc = [=](const float& r, const float& theta, const float& /*phi*/)
			{
				float sqrt3fpi = cl::sycl::sqrt(3.0f / M_PI);
				//float val = 1.0f / 2.0f * sqrt3fpi * cl::sycl::cos(theta + phiii); // Y(l = 1, m = 0)
				float val = 1.0f / 2.0f * sqrt3fpi * cl::sycl::cos(theta); // Y(l = 1, m = 0)
				float result = cl::sycl::fabs(2 * cl::sycl::fabs(val) - r);

				if (result < 0.01f)	// thickness of shell 
					return val < 0 ? -1 : 1;
				else
					return 0;
			};

			// color according to the incoming density
			auto colorFunc = [](const int density)
			{
				if (density > 0)
				{
					return cl::sycl::float4(0, 0, 1, 0); // blue
				}
				else if (density < 0)
				{
					return cl::sycl::float4(1, 1, 0, 0); // yellow
				}
				else
					return  cl::sycl::float4(0, 0, 0, 0); // black
			};

			cl::sycl::float4 finalColor(0.0f, 0.0f, 0.0f, 0.0f);
			cl::sycl::float3 location(0.0f, 0.0f, 0.0f);

			location = camPos + startT * rayDirection;

			float current_t = startT;

			while (current_t < endT)
			{
				location = location + deltaS * rayDirection;
				current_t += deltaS;

				// check if it is inside
				//if (!IsOutside(location))
				float x = location.x();
				float y = location.y();
				float z = location.z();
				//if (x < extent.m_maxX)
				//if ((x < extent.m_maxX) && y < (extent.m_maxY) && (z < extent.m_maxZ) &&
				//	(x > extent.m_minX) && (y > extent.m_minY) && (z > extent.m_minZ))
				//{
					// Convert to spherical coordinated
					//float r = sqrt(location.x*location.x + location.y*location.y + location.z*location.z);

				float r = cl::sycl::length(location);
				float theta = cl::sycl::acos(location.z() / r); // *180 / 3.1415926f; // convert to degrees?
				float phi = cl::sycl::atan2(y, x); // *180 / 3.1415926f;


				cl::sycl::float4 color = colorFunc(densityFunc(r, theta, phi));


				finalColor += color;
				//} // end if check isInside

				// stop the ray, when color reaches the saturation.
				if (finalColor.r() > saturationThreshold || finalColor.g() > saturationThreshold
					|| finalColor.b() > saturationThreshold)
					break;
			}

			// normalizer according to the highest rgb value
			auto normalizer = std::max((float)1.0f, std::max(std::max(finalColor.r(), finalColor.g()), finalColor.b()));
			finalColor /= normalizer;
			finalColor *= 255;


			return cl::sycl::float4(finalColor.r(), finalColor.g(), finalColor.b(), 255.f);
		};
		// END raymarch lambda


		compute_queue.submit([&](cl::sycl::handler& cgh)
		{
			using namespace cl::sycl;

			auto old_lattice = latticeImages[Buffer::Front]->get_access<float4, access::mode::read>(cgh);
			auto new_lattice = latticeImages[Buffer::Back]->get_access<float4, access::mode::write>(cgh);

			sampler periodic{ coordinate_normalization_mode::unnormalized,
							  addressing_mode::none,
							  filtering_mode::nearest };

			auto aspectRatio = (float)old_lattice.get_range()[0] / old_lattice.get_range()[1];
			//float scaleFOV = tan(120.f / 2 * M_PI / 180);
			// scaleFOV?
			cgh.parallel_for<kernels::SphericalHarmonics_Kernel>(range<2>{ old_lattice.get_range() },
				[=, ViewToWorldMtx = m_viewToWorldMtx, camPos = m_vecEye, sphereCenter = glm::vec3(0.f, 0.f, 0.f), sphereRadius2 = 1.96f, raymarch = m_raymarch, deltaS = 0.02f
				](const item<2> i)
			{
				// Minden mehet a regivel, mert jelenleg nem kell az uv koordinate transzformalgatas
				int2 pixelIndex = i.get_id();
				auto getPixelFromOldLattice = [=](int2 in) { return old_lattice.read(in, periodic); };
				auto setPixelForNewLattice = [=](float4 in) { new_lattice.write((int2)i.get_id(), in); };


				glm::vec4 rayVec((2 * (i[0] + 0.5f) / (float)old_lattice.get_range()[0] - 1)* aspectRatio /* * scaleFOV */,
					(1 - 2 * (i[1] + 0.5f) / (float)old_lattice.get_range()[1]) /* * scaleFOV*/,
					-1.0f, 1.0f);

				float t0 = -1E+36f;
				float t1 = -1E+36f;

				glm::vec3 transformedCamRayDir = glm::vec3(ViewToWorldMtx * rayVec) - camPos;
				cl::sycl::float3 transformedCamRayDirFloat3 = cl::sycl::normalize(cl::sycl::float3{ transformedCamRayDir.x, transformedCamRayDir.y, transformedCamRayDir.z });


				auto getIntersections_lambda = [&t0, &t1](const cl::sycl::float3 rayorig, const cl::sycl::float3 raydir, const cl::sycl::float3 sphereCenter,
					const float sphereRadius2) {
					cl::sycl::float3 l = sphereCenter - rayorig;
					float tca = cl::sycl::dot(l, raydir);
					float d2 = cl::sycl::dot(l, l) - tca * tca;

					bool isIntersected = true;
					if ((sphereRadius2 - d2) < 0.0001f) {
						isIntersected = false;

					}
					float thc = cl::sycl::sqrt(sphereRadius2 - d2);

					t0 = tca - thc;
					t1 = tca + thc;

					return isIntersected;

				};

				auto camPosFloat3 = cl::sycl::float3(camPos.x, camPos.y, camPos.z);
				auto bIntersected = getIntersections_lambda(camPosFloat3, transformedCamRayDirFloat3,
					cl::sycl::float3(sphereCenter.x, sphereCenter.y, sphereCenter.z), sphereRadius2);

				cl::sycl::float4 pixelColor;
				if (bIntersected && t0 > 0.0 && t1 > 0.0)
				{
					//pixelColor = cl::sycl::float4(255, 0, 0, 255);
					pixelColor = raymarch(camPosFloat3, transformedCamRayDirFloat3, t0, t1, deltaS);
				}
				// if we are inside the spehere, we trace from the the ray's original position
				else if (bIntersected && t1 > 0.0)
				{
					//pixelColor = cl::sycl::float4(0, 255, 0, 255);
					pixelColor = raymarch(camPosFloat3, transformedCamRayDirFloat3, 0.0, t1, deltaS);
				}
				else
				{
					pixelColor = cl::sycl::float4(0.f, 0.f, 0.f, 255.f);
				}

				// seting rgb value for every pixel
				setPixelForNewLattice(pixelColor);
			});
		});
}