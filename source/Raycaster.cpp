#include <Raycaster.hpp>

using namespace cl::sycl;

Raycaster::Raycaster(std::size_t plat,
    std::size_t dev,
    cl_bitfield type,
    QWindow* parent)
    : InteropWindowImpl(plat, dev, type, parent)
{
}

void Raycaster::mouseDragImpl(QMouseEvent* event_in) {
	phi = (event_in->x() - mousePos.x());
	theta = (event_in->y() - mousePos.y());
}

void Raycaster::resetScene() {
	float xy_lim = 1.0f;
	// For now suposing that the extent is a cube
	 extent = { { { -xy_lim, xy_lim },{ -xy_lim, xy_lim },{ -xy_lim, xy_lim } } };
	 sphereBoundigBox.center = glm::vec3((extent[0][0] + extent[0][1]) / 2, (extent[1][0] + extent[1][1]) / 2, (extent[2][0] + extent[2][1]) / 2);
	 sphereBoundigBox.radius2 = std::powf((extent[0][1] - extent[0][0]) / 2 * cl::sycl::sqrt(3.f), 2);
	 stepSize = (extent[0][1] - extent[0][0]) / 200.f;
}

void Raycaster::mouseWheelEventImpl(QWheelEvent* wheel_event) {
	auto numDegrees = wheel_event->angleDelta() / 8;
	auto x = numDegrees.x();
	auto y = (float)numDegrees.y() / (-50);

	m_vecEye +=  m_vecEye * y;

	needMatrixReset = true;
	if (!getAnimating()) renderNow();
}