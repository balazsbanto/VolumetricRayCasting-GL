#pragma once
// SYCL include
#include <CL/sycl.hpp>
#include <InteropWindowImpl.hpp>

namespace kernels { struct Raycaster_Kernel;}

struct SphereBoundingBox {
	glm::vec3 center{ 0.f, 0.f, 0.f };
	float radius2 = 0.f;
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
	virtual void updateSceneImpl() override;
	virtual void mouseWheelEventImpl(QWheelEvent* wheel_event) override;

protected:
	std::array<std::array<float, 2>, 3 > extent;
	SphereBoundingBox sphereBoundigBox;
	float stepSize = 0.f;

private:
	

};