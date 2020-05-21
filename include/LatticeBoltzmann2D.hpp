#pragma once
// SYCL include
#include <CL/sycl.hpp>
#include <InteropWindowImpl.hpp>

namespace kernels {struct Lbm; }

class LatticeBoltzmann2D : public InteropWindowImpl
{
	Q_OBJECT
public:

	explicit LatticeBoltzmann2D(std::size_t plat,
		std::size_t dev,
		cl_bitfield type,
		QWindow* parent = 0);
	~LatticeBoltzmann2D() = default;

	virtual void mouseDragImpl(QMouseEvent* event_in) override;
	virtual void resetScene() override;
	virtual void updateSceneImpl() override;
	virtual void swapDataBuffers() override;
	void writeOutputsToFile() override;

	void setInput();

private:
	// LBM D2Q9

	//  Distribution Buffers
	std::array <cl::sycl::buffer<float, 1> , 2 > f0_buffers;
	std::array <cl::sycl::buffer<cl::sycl::float4, 1>, 2 > f1234_buffers;
	std::array <cl::sycl::buffer<cl::sycl::float4, 1>, 2 > f5678_buffers;

	// Host vectors
	std::array < std::vector<float>, 2 > f0_host;
	std::array < std::vector<cl::sycl::float4>, 2 > f1234_host;
	std::array < std::vector<cl::sycl::float4>, 2 > f5678_host;
	std::vector<cl::sycl::float2> velocity_host;
	bool* type_host;

	// Output velocity buffer
	cl::sycl::buffer<cl::sycl::float2, 1> velocity_buffer;

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

	// DEBUG
	std::vector<cl::sycl::int2> idVec;
	cl::sycl::buffer<cl::sycl::int2, 1>  idBuffer;
};