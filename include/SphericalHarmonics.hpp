#pragma once
// SYCL include
#include <CL/sycl.hpp>
#include <SYCL-Raycaster.hpp>

namespace kernels { struct SphericalHarmonics_Kernel;}

class SphericalHarmonics : public Raycaster
{
	Q_OBJECT
public:

	explicit SphericalHarmonics(std::size_t plat,
								std::size_t dev,
								cl_bitfield type,
								QWindow* parent = 0);

	~SphericalHarmonics() = default;

	virtual void mouseDragImpl(QMouseEvent* event_in) override;
	virtual void updateSceneImpl() override;
	virtual void mouseWheelEventImpl(QWheelEvent* wheel_event) override;

private:
	
};