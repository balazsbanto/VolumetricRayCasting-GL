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
	virtual virtual void resetScene() override;
	virtual void swapDataBuffers() override;
	void writeOutputsToFile() override;

	void setInput();

private:
	// LBM D2Q9

	//  Distribution Buffers
	std::array < std::unique_ptr<cl::sycl::buffer<float, 1> >, 2 > f0_buffers;
	std::array < std::unique_ptr<cl::sycl::buffer<cl::sycl::float4, 1>>, 2 > f1234_buffers;
	std::array < std::unique_ptr<cl::sycl::buffer<cl::sycl::float4, 1>>, 2 > f5678_buffers;

	// Host vectors
	std::array < std::vector<float>, 2 > f0_host;
	std::array < std::vector<cl::sycl::float4>, 2 > f1234_host;
	std::array < std::vector<cl::sycl::float4>, 2 > f5678_host;
	std::vector<cl::sycl::float2> velocity_host;
	bool* type_host;

	// Output velocity buffer
	std::unique_ptr < cl::sycl::buffer<cl::sycl::float2, 1>> velocity_buffer;

	// 0 - fluid, 1 - boundary
	cl::sycl::buffer<bool, 1>  type_buffer;

	// Unit direction vectors and their buffers
	std::vector<int> h_dirX{ 0, 1, 0, -1,  0, 1, -1,  -1,  1 };
	std::vector<int> h_dirY{ 0, 0, 1,  0, -1, 1,  1,  -1, -1 };

	cl::sycl::buffer<int, 1> h_dirX_buffer;
	cl::sycl::buffer<int, 1> h_dirY_buffer;

	// Weights
	std::vector<float> w{ 4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0 };
	cl::sycl::buffer<float, 1> h_weigt_buffer;

};
