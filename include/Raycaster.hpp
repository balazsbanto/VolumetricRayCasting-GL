#pragma once
// SYCL include
#include <CL/sycl.hpp>
#include <InteropWindowImpl.hpp>
#include <Common.hpp>

using namespace cl::sycl;

namespace kernels { struct Raycaster_Kernel;}

const auto isInside = [](const std::array<std::array<float, 2>, 3 >& extent, const float3& location) {
	return location.get_value(X) >= extent[0][0] && location.get_value(X) <= extent[0][1]
		&& location.get_value(Y) >= extent[1][0] && location.get_value(Y) <= extent[1][1]
		&& location.get_value(Z) >= extent[2][0] && location.get_value(Z) <= extent[2][1];

};

struct SphereIntersection {
	bool isIntersected = false;
	float t0 = -1E+36f;
	float t1 = -1E+36f;

};

struct SphereBoundingBox {
	glm::vec3 center{ 0.f, 0.f, 0.f };
	float radius2 = 0.f;
};

auto getIntersections = [](const float3& rayorig, const float3& raydir, const SphereBoundingBox& boundingBox) {
	const float3 sphereCenter{ boundingBox.center.x,  boundingBox.center.y,  boundingBox.center.z };
	float3 l = sphereCenter - rayorig;
	float tca = cl::sycl::dot(l, raydir);
	float d2 = cl::sycl::dot(l, l) - tca * tca;

	SphereIntersection spherIntersection;
	if ((boundingBox.radius2 - d2) < 0.0001f) {
		spherIntersection.isIntersected = false;

	}
	else {
		float thc = cl::sycl::sqrt(boundingBox.radius2 - d2);
		spherIntersection.isIntersected = true;
		spherIntersection.t0 = tca - thc;
		spherIntersection.t1 = tca + thc;
	}

	return spherIntersection;

};

class Raycaster : public InteropWindowImpl
{
	Q_OBJECT
public:

	explicit Raycaster(std::size_t plat,
								std::size_t dev,
								cl_bitfield type,
								QWindow* parent = 0);

	~Raycaster() = default;

	virtual virtual void resetScene() override;
	virtual void mouseDragImpl(QMouseEvent* event_in) override;
	virtual void updateSceneImpl() override = 0;
	virtual void mouseWheelEventImpl(QWheelEvent* wheel_event) override;

protected:
	std::array<std::array<float, 2>, 3 > extent;
	SphereBoundingBox sphereBoundigBox;
	float stepSize = 0.f;

private:
	
};