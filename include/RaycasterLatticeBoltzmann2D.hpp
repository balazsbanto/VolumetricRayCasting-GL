#pragma once
#include <Raycaster.hpp>

class RaycasterLatticeBoltzmann2D : public Raycaster {
	Q_OBJECT
public:
	explicit RaycasterLatticeBoltzmann2D(std::size_t plat,
		std::size_t dev,
		cl_bitfield type,
		QWindow* parent = 0);

	~RaycasterLatticeBoltzmann2D() = default;

	virtual void updateSceneImpl() override;

};