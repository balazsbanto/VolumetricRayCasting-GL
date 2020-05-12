#pragma once
// SYCL include
#include <CL/sycl.hpp>
#include <InteropWindowImpl.hpp>

namespace kernels { struct SphericalHarmonics_Kernel;}

struct SphereBoundingBox {
	glm::vec3 center{ 0.f, 0.f, 0.f };
	float radius2 = 0.f;
};

class SphericalHarmonics : public InteropWindowImpl
{
	Q_OBJECT
public:

	explicit SphericalHarmonics(std::size_t plat,
								std::size_t dev,
								cl_bitfield type,
								QWindow* parent = 0);

	~SphericalHarmonics() = default;

	virtual virtual void resetScene() override;
	virtual void mouseDragImpl(QMouseEvent* event_in) override;
	virtual void updateSceneImpl() override;
	virtual void mouseWheelEventImpl(QWheelEvent* wheel_event) override;

protected:
	std::array<std::array<float, 2>, 3 > extent;
	SphereBoundingBox sphereBoundigBox;
	float stepSize = 0.f;

private:
	

};