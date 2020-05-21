#pragma once
#include <Raycaster.hpp>

// LBM D3Q19
class RaycasterLbm3D : public Raycaster {
	Q_OBJECT
public:
	explicit RaycasterLbm3D(std::size_t plat,
		std::size_t dev,
		cl_bitfield type,
		QWindow* parent = 0);

	~RaycasterLbm3D() = default;

	virtual void updateSceneImpl() override;
	virtual void resetScene() override;
	virtual void swapDataBuffers() override;
	void writeOutputsToFile() override;

	void setInput();
	cl::sycl::int3 getMeshDim();
	//int getMeshSize();

private:
	cl::sycl::int3 meshDim = { 0, 0, 0 };
	//  Distribution Buffers
	std::array < cl::sycl::buffer<float, 1>, 2 > f0_buffers;
	std::array < cl::sycl::buffer<cl::sycl::float4, 1>, 2 > f1to4_buffers;
	std::array < cl::sycl::buffer<cl::sycl::float2, 1>, 2 > f56_buffers;
	std::array < cl::sycl::buffer<cl::sycl::float8, 1>, 2 > f7to14_buffers;
	std::array < cl::sycl::buffer<cl::sycl::float4, 1>, 2 > f15to18_buffers;
	cl::sycl::buffer<cl::sycl::float3, 1> velocity_buffer;
	cl::sycl::buffer<bool, 1>  type_buffer; // 0 - fluid, 1 - boundary

	// Host vectors
	std::array < std::vector<float>, 2 > f0_host;
	std::array < std::vector<cl::sycl::float4>, 2 > f1to4_host;
	std::array < std::vector<cl::sycl::float2>, 2 > f56_host;
	std::array < std::vector<cl::sycl::float8>, 2 > f7to14_host;
	std::array < std::vector<cl::sycl::float4>, 2 > f15to18_host;
	std::vector<cl::sycl::float3> velocity_host;
	bool* type_host;
};