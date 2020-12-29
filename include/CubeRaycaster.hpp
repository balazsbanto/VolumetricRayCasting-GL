#pragma once
#include <Raycaster.hpp>

class CubeRaycaster : public Raycaster {
	Q_OBJECT
public:
	explicit CubeRaycaster(std::size_t plat,
		std::size_t dev,
		cl_bitfield type,
		QWindow* parent = 0);

	~CubeRaycaster() = default;

	virtual void updateSceneImpl() override;

};