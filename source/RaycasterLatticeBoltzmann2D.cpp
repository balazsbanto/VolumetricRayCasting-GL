#include <RaycasterLatticeBoltzmann2D.hpp>
#include <iomanip>
#include <Common.hpp>
#include <Lbm2DCommon.hpp>

namespace kernels { struct Raycaster_LBM2D; }
using namespace cl::sycl;

// Start equilibrium distribution for the current initial config (D2Q9, rho = 10) 
const float F0_EQ = 4.4444444444f;
const float F1234_EQ = 1.1111111111f;
const float F5678_EQ = 0.2777777777f;

const auto transformWorldCoordinates = [](const float3& worldLocation, const int extentLim, const int lbmSize ) {
	return float3{ worldLocation.get_value(X) + extentLim, extentLim - worldLocation.get_value(Y), worldLocation.get_value(Z) } *(lbmSize / (2 * extentLim));
};


const auto densityFunc = [](const float3& lbmSpaceCoordinates)
{
	return 1;
};

// color according to the incoming density
//const auto colorFunc = [](const int density)
//{
//	if (density > 0)
//	{
//		return float4(0, 0, 1, 1); // blue
//	}
//	else if (density < 0)
//	{
//		return float4(1, 1, 0, 1); // yellow
//	}
//	else
//		return  float4(0, 0, 0, 1); // black
//};

template <cl::sycl::access::target Target>
struct Lbm2DSpaceAccessors {
	const DistributionBuffers<Target, cl::sycl::access::mode::discard_write> &distributions;
	const cl::sycl::accessor<cl::sycl::float2, 1, cl::sycl::access::mode::discard_write, Target,
		cl::sycl::access::placeholder::false_t> &velocity;
	const cl::sycl::accessor<bool, 1, cl::sycl::access::mode::read, Target,
		cl::sycl::access::placeholder::false_t> &cellType;


};

template <cl::sycl::access::target Target>
struct raymarch {
	float4 operator()(const float3& camPos, const float3& rayDirection, const float startT, const float endT,
	const float stepSize, const std::array<std::array<float, 2>, 3 >& extent, const ScreenSize& screenSize,
	const DistributionBuffers<Target, access::mode::read>& inDistributionBuffers,
	const Lbm2DSpaceAccessors<Target>& spaceAccessors
#ifdef RUN_ON_CPU
	, std::ofstream& rayPointsFile
#endif // RUN_ON_CPU
	) const
{
	int saturationThreshold = 0;

	float4 finalColor(0.0f, 0.0f, 0.0f, 0.0f);
	float3 location(0.0f, 0.0f, 0.0f);

	location = camPos + startT * rayDirection;

	float current_t = startT;

	bool isSaturated = false;

	while (current_t < endT && !isSaturated)
	{
		location += stepSize * rayDirection;
		current_t += stepSize;

		if (isInside(extent, location)) {

			//float4 color(0.0f, 0.0f, 0.0f, 0.0f);
			auto lbmSpaceCoordinates = transformWorldCoordinates(location, 1, screenSize.width);
			int2 id{ int(lbmSpaceCoordinates.get_value(X)), int(lbmSpaceCoordinates.get_value(Y)) };

			int pos = id.get_value(X) + screenSize.width * id.get_value(Y);

			auto cellAfterCollision = collide(Distributions{ inDistributionBuffers.f0[pos], inDistributionBuffers.f1234[pos],
				inDistributionBuffers.f5678[pos] }, spaceAccessors.cellType[pos]);

			streamToNeighbours<Target >()(id, pos, screenSize, cellAfterCollision.distributions, spaceAccessors.distributions);

			spaceAccessors.velocity[pos] = cellAfterCollision.velocity;

			float4 color = colorFunc(cellAfterCollision.velocity, cellAfterCollision.cellType);
			//float4 color = colorFunc(densityFunc(transformWorldCoordinates(location)));

			finalColor += color;

			//isSaturated = finalColor.r() > saturationThreshold || finalColor.g() > saturationThreshold || finalColor.b() > saturationThreshold;
			isSaturated = true;

#ifdef RUN_ON_CPU
			//if (cl::sycl::fabs(location.get_value(Z)) < stepSize) {
			auto transformedToLbm = transformWorldCoordinates(location, 1, screenSize.width);
			rayPointsFile << (int)transformedToLbm.get_value(X) << " " << (int)transformedToLbm.get_value(Y) << " " << transformedToLbm.get_value(Z) << "\n";
			//}
#endif
		}
	}

	// normalizer according to the highest rgb value
	auto normalizer = std::max(1.0f, std::max(std::max(finalColor.r(), finalColor.g()), finalColor.b()));
	finalColor /= normalizer;
	finalColor.set_value(W, 1.f);

	return finalColor;
}
};

RaycasterLatticeBoltzmann2D::RaycasterLatticeBoltzmann2D(std::size_t plat,
    std::size_t dev,
    cl_bitfield type,
    QWindow* parent)
    : Raycaster(plat, dev, type, parent)
{
}

#ifdef RUN_ON_CPU
void RaycasterLatticeBoltzmann2D::updateSceneImpl() {

	auto if0 = f0_buffers[Buffer::Front]->get_access<access::mode::read>();
	auto if1234 = f1234_buffers[Buffer::Front]->get_access<access::mode::read>();
	auto if5678 = f5678_buffers[Buffer::Front]->get_access<access::mode::read>();
	auto type = type_buffer.get_access<access::mode::read>();

	// Output
	auto of0 = f0_buffers[Buffer::Back]->get_access<access::mode::discard_write>();
	auto of1234 = f1234_buffers[Buffer::Back]->get_access<access::mode::discard_write>();
	auto of5678 = f5678_buffers[Buffer::Back]->get_access<access::mode::discard_write>();
	auto velocity_out = velocity_buffer->get_access<access::mode::discard_write>();


	int screen_width = width();
	int screen_height = height();
	const float aspectRatio = (float)screen_width / screen_height;
	auto ViewToWorldMtx = m_viewToWorldMtx;
	auto camPosGlm = m_vecEye;

	std::ofstream rayPointsFile("rayPointsFile.txt");

	for (int y = 0; y < screen_height; y++) {
		for (int x = 0; x < screen_width; x++) {

			int2 i{ x, y };

			glm::vec4 rayVec((2 * (i.get_value(0) + 0.5f) / (float)screenSize.width - 1) * aspectRatio /* * scaleFOV */,
				(1 - 2 * (i.get_value(1) + 0.5f) / (float)screenSize.height) /* * scaleFOV*/,
				-1.0f, 1.0f);

			// Quick switch to glm vectors to perform 4x4 matrix x vector multiplication, since SYCL has not have yet these operation
			glm::vec3 transformedCamRayDirGlm = glm::vec3(ViewToWorldMtx * rayVec) - camPosGlm;
			float3 normalizedCamRayDir = cl::sycl::normalize(float3{ transformedCamRayDirGlm.x, transformedCamRayDirGlm.y, transformedCamRayDirGlm.z });

			auto cameraPos = float3(camPosGlm.x, camPosGlm.y, camPosGlm.z);
			auto spherIntersection = getIntersections(cameraPos, normalizedCamRayDir, sphereBoundigBox);

			float4 pixelColor;
			if (spherIntersection.isIntersected && spherIntersection.t0 > 0.0 && spherIntersection.t1 > 0.0)
			{
				pixelColor = raymarch<access::target::host_buffer>()
					(cameraPos, normalizedCamRayDir, spherIntersection.t0, spherIntersection.t1, stepSize, extent, screenSize,
						DistributionBuffers<access::target::host_buffer, access::mode::read>{ if0, if1234, if5678 },
						Lbm2DSpaceAccessors< access::target::host_buffer > {
					DistributionBuffers<access::target::host_buffer, access::mode::discard_write>{ of0, of1234, of5678 },
						velocity_out, type },
						rayPointsFile
				);
			}
			// if we are inside the spehere, we trace from the the ray's original position
			else if (spherIntersection.isIntersected && spherIntersection.t1 > 0.f)
			{
				pixelColor = raymarch<access::target::host_buffer>()
					(cameraPos, normalizedCamRayDir, 0.0, spherIntersection.t1, stepSize, extent, screenSize,
						DistributionBuffers<access::target::host_buffer, access::mode::read>{ if0, if1234, if5678 },
						Lbm2DSpaceAccessors< access::target::host_buffer > {
					DistributionBuffers<access::target::host_buffer, access::mode::discard_write>{ of0, of1234, of5678 },
						velocity_out, type },
						rayPointsFile
				);
			}
			else
			{
				pixelColor = float4(0.f, 0.f, 0.f, 1.f);
			}

		}
	}
	rayPointsFile.close();
}
#else
void RaycasterLatticeBoltzmann2D::updateSceneImpl() {

	auto screenSize = ScreenSize{ width(), height() };

	const float aspectRatio = screenSize.aspectRatio();

	compute_queue.submit([&](cl::sycl::handler& cgh)
	{
		// Input buffers
		auto if0 = f0_buffers[Buffer::Front]->get_access<access::mode::read>(cgh);
		auto if1234 = f1234_buffers[Buffer::Front]->get_access<access::mode::read>(cgh);
		auto if5678 = f5678_buffers[Buffer::Front]->get_access<access::mode::read>(cgh);
		auto type = type_buffer.get_access<access::mode::read>(cgh);

		// Output buffers
		auto velocity_out = velocity_buffer->get_access<access::mode::discard_write>(cgh);
		auto of0 = f0_buffers[Buffer::Back]->get_access<access::mode::discard_write>(cgh);
		auto of1234 = f1234_buffers[Buffer::Back]->get_access<access::mode::discard_write>(cgh);
		auto of5678 = f5678_buffers[Buffer::Back]->get_access<access::mode::discard_write>(cgh);

		auto new_lattice = latticeImages[Buffer::Back]->get_access<float4, access::mode::discard_write>(cgh);


		cgh.parallel_for<kernels::Raycaster_LBM2D>(range<2>{ new_lattice.get_range() },
			[=, ViewToWorldMtx = m_viewToWorldMtx, camPosGlm = m_vecEye, sphereBoundigBox = sphereBoundigBox, stepSize = stepSize,
			 extent = extent](const item<2> i)
		{

			glm::vec4 rayVec((2 * (i[0] + 0.5f) / (float)screenSize.width - 1) * aspectRatio /* * scaleFOV */,
				(1 - 2 * (i[1] + 0.5f) / (float)screenSize.height) /* * scaleFOV*/,
				-1.0f, 1.0f);

			// Quick switch to glm vectors to perform 4x4 matrix x vector multiplication, since SYCL has not have yet these operation
			glm::vec3 transformedCamRayDirGlm = glm::vec3(ViewToWorldMtx * rayVec) - camPosGlm;
			float3 normalizedCamRayDir = cl::sycl::normalize(float3{ transformedCamRayDirGlm.x, transformedCamRayDirGlm.y, transformedCamRayDirGlm.z });

			auto cameraPos = float3(camPosGlm.x, camPosGlm.y, camPosGlm.z);
			auto spherIntersection = getIntersections(cameraPos, normalizedCamRayDir, sphereBoundigBox);

			float4 pixelColor;
			if (spherIntersection.isIntersected && spherIntersection.t0 > 0.0 && spherIntersection.t1 > 0.0)
			{
				pixelColor = raymarch<access::target::global_buffer>()
					(cameraPos, normalizedCamRayDir, spherIntersection.t0, spherIntersection.t1, stepSize, extent, screenSize,
						DistributionBuffers<access::target::global_buffer, access::mode::read>{ if0, if1234, if5678 },
						Lbm2DSpaceAccessors< access::target::global_buffer > {
							DistributionBuffers<access::target::global_buffer, access::mode::discard_write>{ of0, of1234, of5678 },
								velocity_out, type }
					);
			}
			// if we are inside the spehere, we trace from the the ray's original position
			else if (spherIntersection.isIntersected && spherIntersection.t1 > 0.f)
			{
				pixelColor = raymarch<access::target::global_buffer>()
					(cameraPos, normalizedCamRayDir, 0.0, spherIntersection.t1, stepSize, extent, screenSize,
						DistributionBuffers<access::target::global_buffer, access::mode::read>{ if0, if1234, if5678 },
						Lbm2DSpaceAccessors< access::target::global_buffer > {
							DistributionBuffers<access::target::global_buffer, access::mode::discard_write>{ of0, of1234, of5678 },
							velocity_out, type }
					);
			}
			else
			{
				pixelColor = float4(0.f, 0.f, 0.f, 1.f);
			}

			auto setPixelForNewLattice = [=](float4 in) { new_lattice.write((int2)i.get_id(), in); };
			// seting rgb value for every pixel
			setPixelForNewLattice(pixelColor);
		});
	});
}
#endif

void RaycasterLatticeBoltzmann2D::resetScene() {

	using namespace cl::sycl;

	// Initial velocity is 0
	type_host = new bool[getMeshSize()];
	f0_host[Buffer::Front] = std::vector<float>(getMeshSize(), F0_EQ);
	f1234_host[Buffer::Front] = std::vector<float4>(getMeshSize(), float4{ F1234_EQ });
	f5678_host[Buffer::Front] = std::vector<float4>(getMeshSize(), float4{ F5678_EQ });

	f0_host[Buffer::Back] = f0_host[Buffer::Front];
	f1234_host[Buffer::Back] = f1234_host[Buffer::Front];
	f5678_host[Buffer::Back] = f5678_host[Buffer::Front];

	f0_buffers[Buffer::Front] = std::make_unique<buffer<float, 1>>(f0_host[Buffer::Front].data(), range<1> {getMeshSize()});
	f1234_buffers[Buffer::Front] = std::make_unique<buffer<float4, 1>>(f1234_host[Buffer::Front].data(), range<1> {getMeshSize()});
	f5678_buffers[Buffer::Front] = std::make_unique<buffer<float4, 1>>(f5678_host[Buffer::Front].data(), range<1> { getMeshSize()});


	f0_buffers[Buffer::Back] = std::make_unique<buffer<float, 1>>(f0_host[Buffer::Back].data(), range<1> {getMeshSize()});
	f1234_buffers[Buffer::Back] = std::make_unique<buffer<float4, 1>>(f1234_host[Buffer::Back].data(), range<1> {getMeshSize()});
	f5678_buffers[Buffer::Back] = std::make_unique<buffer<float4, 1>>(f5678_host[Buffer::Back].data(), range<1> { getMeshSize()});

	velocity_host = std::vector<float2>(getMeshSize(), float2{ 0.f, 0.f });
	velocity_buffer = std::make_unique< buffer<float2, 1>>(velocity_host.data(), range<1> { getMeshSize()});

	// Vector with contants
	h_dirX_buffer = buffer<int, 1>{ h_dirX.data(), range<1> {h_dirX.size()} };
	h_dirY_buffer = buffer<int, 1>{ h_dirY.data(), range<1> {h_dirY.size()} };
	h_weigt_buffer = buffer<float, 1>{ w.data(), range<1> {w.size()} };


	for (int y = 0; y < height(); y++) {
		for (int x = 0; x < width(); x++) {

			int pos = x + y * width();

			// Initialize boundary cells
			if (x == 0 || x == (width() - 1) || y == 0 || y == (height() - 1))
			{
				type_host[pos] = true;
			}

			// Initialize fluid cells
			else
			{
				type_host[pos] = false;
			}

		}
	}

	type_buffer = buffer<bool, 1>{ type_host, range<1> {getMeshSize()} };

	writeOutputsToFile();
	setInput();
	writeOutputsToFile();
}

void RaycasterLatticeBoltzmann2D::setInput() {
	// Set a test velocity of { 0.4395f, 0.4395f } to (64, 10)
	using namespace cl::sycl;
	int x = 64;
	int y = height() - 1 - 10;
	int pos = x + width() * y;

	auto if0 = f0_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read_write>();
	auto if1234 = f1234_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read_write>();
	auto if5678 = f5678_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read_write>();

	auto velocity_out = velocity_buffer->get_access<access::mode::read>();

	// Calculate density from input distribution
	float rho = if0[pos]
		+ if1234[pos].get_value(0) + if1234[pos].get_value(1) + if1234[pos].get_value(2) + if1234[pos].get_value(3) +
		+if5678[pos].get_value(0) + if5678[pos].get_value(1) + if5678[pos].get_value(2) + if5678[pos].get_value(3);

	// Increase the speed by input speed
	//velocity_out[pos] += dragVelocity;

	float2 newVel = velocity_out[pos] + float2{ 0.4395f, 0.4395f };;

	// Calculate new distribution based on input speed

	if0[pos] = computefEq(rho, w[0], cl::sycl::float2{ h_dirX[0], h_dirY[0] }, newVel);

	if1234[pos].set_value(0, computefEq(rho, w[1], cl::sycl::float2{ h_dirX[1], h_dirY[1] }, newVel));
	if1234[pos].set_value(1, computefEq(rho, w[2], cl::sycl::float2{ h_dirX[2], h_dirY[2] }, newVel));
	if1234[pos].set_value(2, computefEq(rho, w[3], cl::sycl::float2{ h_dirX[3], h_dirY[3] }, newVel));
	if1234[pos].set_value(3, computefEq(rho, w[4], cl::sycl::float2{ h_dirX[4], h_dirY[4] }, newVel));

	if5678[pos].set_value(0, computefEq(rho, w[5], cl::sycl::float2{ h_dirX[5], h_dirY[5] }, newVel));
	if5678[pos].set_value(1, computefEq(rho, w[6], cl::sycl::float2{ h_dirX[6], h_dirY[6] }, newVel));
	if5678[pos].set_value(2, computefEq(rho, w[7], cl::sycl::float2{ h_dirX[7], h_dirY[7] }, newVel));
	if5678[pos].set_value(3, computefEq(rho, w[8], cl::sycl::float2{ h_dirX[8], h_dirY[8] }, newVel));
}

void RaycasterLatticeBoltzmann2D::swapDataBuffers() {
	std::swap(f0_buffers[Buffer::Front], f0_buffers[Buffer::Back]);
	std::swap(f1234_buffers[Buffer::Front], f1234_buffers[Buffer::Back]);
	std::swap(f5678_buffers[Buffer::Front], f5678_buffers[Buffer::Back]);
}

void RaycasterLatticeBoltzmann2D::writeOutputsToFile() {
#ifndef WRITE_OUTPUT_TO_FILE
	return;
#endif // !WRITE_OUTPUT_TO_FILE

	static int fileIndex = 0;

	auto f0 = f0_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read>();
	auto f1234 = f1234_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read>();
	auto f5678 = f5678_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read>();
	auto velocity = velocity_buffer->get_access<cl::sycl::access::mode::read>();

	std::ofstream of0_file("of0_" + std::to_string(fileIndex) + ".txt");
	std::ofstream of1234_file("of1234_" + std::to_string(fileIndex) + ".txt");
	std::ofstream of5678_file("of5678_" + std::to_string(fileIndex) + ".txt");
	std::ofstream velocity_file("velocity_" + std::to_string(fileIndex) + ".txt");


	for (int i = 0; i < f0.get_count(); i++) {
		of0_file << std::setprecision(5) << (float)f0[i] << "\n";
	}

	for (int i = 0; i < f1234.get_count(); i++) {
		//qDebug() << f1234[i].get_value(0) << "\n";
		of1234_file << std::setprecision(5) << f1234[i].get_value(0) << "\t" << f1234[i].get_value(1) << "\t" << f1234[i].get_value(2) << "\t" << f1234[i].get_value(3) << "\n";

	}

	for (int i = 0; i < f5678.get_count(); i++) {
		of5678_file << std::setprecision(5) << f5678[i].get_value(0) << "\t" << f5678[i].get_value(1) << "\t" << f5678[i].get_value(2) << "\t" << f5678[i].get_value(3) << "\n";
	}

	for (int i = 0; i < velocity.get_count(); i++) {
		velocity_file << std::setprecision(5) << velocity[i].get_value(0) << "\t" << velocity[i].get_value(1) << "\n";
	}


	of0_file.close();
	of1234_file.close();
	of5678_file.close();
	velocity_file.close();

	fileIndex++;
}