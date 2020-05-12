#pragma once
#include <Raycaster.hpp>

class SphericalHarmonicsRaycaster : public Raycaster {
	Q_OBJECT
public:
	explicit SphericalHarmonicsRaycaster(std::size_t plat,
		std::size_t dev,
		cl_bitfield type,
		QWindow* parent = 0);

	~SphericalHarmonicsRaycaster() = default;

	virtual void updateSceneImpl() override;

};