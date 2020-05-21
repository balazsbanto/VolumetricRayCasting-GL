#include <LatticeBoltzmann2D.hpp>
#include <iomanip>
#include <Lbm2DCommon.hpp>

// Start equilibrium distribution for the current initial config (D2Q9, rho = 10) 
const float F0_EQ = 4.4444444444f;
const float F1234_EQ = 1.1111111111f;
const float F5678_EQ = 0.2777777777f;


LatticeBoltzmann2D::LatticeBoltzmann2D(std::size_t plat,
    std::size_t dev,
    cl_bitfield type,
    QWindow* parent)
    : InteropWindowImpl(plat, dev, type, parent)
{
}

void LatticeBoltzmann2D::mouseDragImpl(QMouseEvent* event_in) {
	using namespace cl::sycl;

	/*phi = (event_in->x() - mousePos.x());
	theta = (event_in->y() - mousePos.y());*/

	float2 dragVelocity{ event_in->x() - mousePos.x(),  mousePos.y() - event_in->y() };
	auto magnitude = length(dragVelocity);
	dragVelocity /= (1 + 2 * magnitude);

	// Set new distributions
	int x = event_in->x();
	int y = height() - 1 - event_in->y();
	int pos = x + width() * y;

	auto if0 = f0_buffers[Buffer::Front].get_access<cl::sycl::access::mode::read_write>();
	auto if1234 = f1234_buffers[Buffer::Front].get_access<cl::sycl::access::mode::read_write>();
	auto if5678 = f5678_buffers[Buffer::Front].get_access<cl::sycl::access::mode::read_write>();

	auto velocity_out = velocity_buffer.get_access<access::mode::read>();

	// Calculate density from input distribution
	float rho = if0[pos]
		+ if1234[pos].get_value(0) + if1234[pos].get_value(1) + if1234[pos].get_value(2) + if1234[pos].get_value(3) +
		+if5678[pos].get_value(0) + if5678[pos].get_value(1) + if5678[pos].get_value(2) + if5678[pos].get_value(3);


	float2 newVel = velocity_out[pos] + dragVelocity;

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

	/*qDebug() << "event_in: " << event_in->x() << " " << event_in->y();
	qDebug() << "mousePos: " << mousePos.x() << " " <<  mousePos.y() ;
	qDebug() << "phi: " << phi ;
	qDebug() << "theta: " << theta;*/
}

void LatticeBoltzmann2D::resetScene() {

	using namespace cl::sycl;

	idVec = std::vector<int2>(getNrOfPixels(), int2{ -1, -1 });
	idBuffer = buffer<int2, 1>(idVec.data(), range<1> { getNrOfPixels()});
	// Initial velocity is 0
	type_host = new bool[getNrOfPixels()];
	f0_host[Buffer::Front] = std::vector<float>(getNrOfPixels(), F0_EQ);
	f1234_host[Buffer::Front] = std::vector<float4>(getNrOfPixels(), float4{ F1234_EQ });
	f5678_host[Buffer::Front] = std::vector<float4>(getNrOfPixels(), float4{ F5678_EQ });

	f0_host[Buffer::Back] = f0_host[Buffer::Front];
	f1234_host[Buffer::Back] = f1234_host[Buffer::Front];
	f5678_host[Buffer::Back] = f5678_host[Buffer::Front];

	f0_buffers[Buffer::Front] = buffer<float, 1>(f0_host[Buffer::Front].data(), range<1> {getNrOfPixels()});
	f1234_buffers[Buffer::Front] = buffer<float4, 1>(f1234_host[Buffer::Front].data(), range<1> {getNrOfPixels()});
	f5678_buffers[Buffer::Front] = buffer<float4, 1>(f5678_host[Buffer::Front].data(), range<1> { getNrOfPixels()});


	f0_buffers[Buffer::Back] = buffer<float, 1>(f0_host[Buffer::Back].data(), range<1> {getNrOfPixels()});
	f1234_buffers[Buffer::Back] = buffer<float4, 1>(f1234_host[Buffer::Back].data(), range<1> {getNrOfPixels()});
	f5678_buffers[Buffer::Back] = buffer<float4, 1>(f5678_host[Buffer::Back].data(), range<1> { getNrOfPixels()});

	velocity_host = std::vector<float2>(getNrOfPixels(), float2{ 0.f, 0.f });
	velocity_buffer =  buffer<float2, 1>(velocity_host.data(), range<1> { getNrOfPixels()});

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

	type_buffer = buffer<bool, 1>{ type_host, range<1> {getNrOfPixels()} };

	writeOutputsToFile();
	setInput();
	writeOutputsToFile();
}

void LatticeBoltzmann2D::setInput() {

	// Set a test velocity of { 0.4395f, 0.4395f } to (64, 10)
	using namespace cl::sycl;
	int x = width() / 2;
	int y = height() / 2;
	int pos = x + width() * y;

	auto if0 = f0_buffers[Buffer::Front].get_access<cl::sycl::access::mode::read_write>();
	auto if1234 = f1234_buffers[Buffer::Front].get_access<cl::sycl::access::mode::read_write>();
	auto if5678 = f5678_buffers[Buffer::Front].get_access<cl::sycl::access::mode::read_write>();

	auto velocity_out = velocity_buffer.get_access<access::mode::read>();

	// Calculate density from input distribution
	float rho = if0[pos]
		+ if1234[pos].get_value(0) + if1234[pos].get_value(1) + if1234[pos].get_value(2) + if1234[pos].get_value(3) +
		+if5678[pos].get_value(0) + if5678[pos].get_value(1) + if5678[pos].get_value(2) + if5678[pos].get_value(3);

	// Increase the speed by input speed
	//velocity_out[pos] += dragVelocity;

	float2 newVel = velocity_out[pos] + float2{ 1.f, 1.f };

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

void LatticeBoltzmann2D::writeOutputsToFile() {
#ifndef WRITE_OUTPUT_TO_FILE
	return;
#endif // !WRITE_OUTPUT_TO_FILE
	static int fileIndex = 0;

	auto f0 = f0_buffers[Buffer::Front].get_access<cl::sycl::access::mode::read>();
	auto f1234 = f1234_buffers[Buffer::Front].get_access<cl::sycl::access::mode::read>();
	auto f5678 = f5678_buffers[Buffer::Front].get_access<cl::sycl::access::mode::read>();
	auto velocity = velocity_buffer.get_access<cl::sycl::access::mode::read>();

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

	auto pd = idBuffer.get_access<cl::sycl::access::mode::read>();

	std::ofstream posID("posId.txt");
	for (int i = 0; i < pd.get_count(); i++) {
		posID << pd[i].get_value(0) << " " << pd[i].get_value(1) << "\n";
	}
	posID.close();

}
#ifdef RUN_ON_CPU
void LatticeBoltzmann2D::updateSceneImpl() {
	using namespace cl::sycl;

	auto if0 = f0_buffers[Buffer::Front].get_access<access::mode::read>();
	auto if1234 = f1234_buffers[Buffer::Front].get_access<access::mode::read>();
	auto if5678 = f5678_buffers[Buffer::Front].get_access<access::mode::read>();
	auto type = type_buffer.get_access<access::mode::read>();

	// Output
	auto of0 = f0_buffers[Buffer::Back].get_access<access::mode::discard_write>();
	auto of1234 = f1234_buffers[Buffer::Back].get_access<access::mode::discard_write>();
	auto of5678 = f5678_buffers[Buffer::Back].get_access<access::mode::discard_write>();
	auto velocity_out = velocity_buffer.get_access<access::mode::discard_write>();


	for (int y = 0; y < screenSize.height; y++) {
		for (int x = 0; x < screenSize.width; x++) {

			int2 id(x, y);
			int pos = id.get_value(0) + screenSize.width * id.get_value(1);

			auto cellAfterCollision = collide(Distributions{ if0[pos], if1234[pos], if5678[pos] }, type[pos]);

			streamToNeighbours< access::target::host_buffer >()(id, pos, screenSize, cellAfterCollision.distributions
				, DistributionBuffers<access::target::host_buffer, access::mode::discard_write>{ of0, of1234, of5678 });

			velocity_out[pos] = cellAfterCollision.velocity;

			auto finalPixelColor = colorFunc(cellAfterCollision.velocity, cellAfterCollision.cellType);


			/*if (finalPixelColor.get_value(0) > 0  || finalPixelColor.get_value(1) > 0 || finalPixelColor.get_value(2) > 0 )
				qDebug() << finalPixelColor.get_value(0) << " " << finalPixelColor.get_value(1) << " " << finalPixelColor.get_value(2) << finalPixelColor.get_value(3) << "\n";*/

		}
	}	
}
#else
void LatticeBoltzmann2D::updateSceneImpl() {

	using namespace cl::sycl;

	compute_queue.submit([&](cl::sycl::handler& cgh)
	{
		// Input buffers
		auto if0 = f0_buffers[Buffer::Front].get_access<access::mode::read>(cgh);
		auto if1234 = f1234_buffers[Buffer::Front].get_access<access::mode::read>(cgh);
		auto if5678 = f5678_buffers[Buffer::Front].get_access<access::mode::read>(cgh);
		auto type = type_buffer.get_access<access::mode::read>(cgh);

		// Output buffers
		auto velocity_out = velocity_buffer.get_access<access::mode::discard_write>(cgh);
		auto of0 = f0_buffers[Buffer::Back].get_access<access::mode::discard_write>(cgh);
		auto of1234 = f1234_buffers[Buffer::Back].get_access<access::mode::discard_write>(cgh);
		auto of5678 = f5678_buffers[Buffer::Back].get_access<access::mode::discard_write>(cgh);

		auto new_lattice = latticeImages[Buffer::Back]->get_access<float4, access::mode::discard_write>(cgh);

		auto idAcc = idBuffer.get_access<access::mode::discard_write>(cgh);

		cgh.parallel_for<kernels::Lbm>(range<2>{ new_lattice.get_range() },
			[=, screenSize = screenSize](const item<2> i)
		{

			int2 id = (int2)i.get_id();

			int pos = id.get_value(0) + screenSize.width * id.get_value(1);
			idAcc[pos] = id;

			auto cellAfterCollision = collide(Distributions{ if0[pos], if1234[pos], if5678[pos] }, type[pos]);

			streamToNeighbours< access::target::global_buffer >()(id, pos, screenSize, cellAfterCollision.distributions
				, DistributionBuffers<access::target::global_buffer,
					access::mode::discard_write>{ of0, of1234, of5678 });

			velocity_out[pos] = cellAfterCollision.velocity;
			auto finalPixelColor = colorFunc(cellAfterCollision.velocity, cellAfterCollision.cellType);

			auto setPixelForNewLattice = [=](float4 in) { new_lattice.write(id, in); };
			setPixelForNewLattice(finalPixelColor);

		});
	});
}
#endif


void LatticeBoltzmann2D::swapDataBuffers() {
	std::swap(f0_buffers[Buffer::Front], f0_buffers[Buffer::Back]);
	std::swap(f1234_buffers[Buffer::Front], f1234_buffers[Buffer::Back]);
	std::swap(f5678_buffers[Buffer::Front], f5678_buffers[Buffer::Back]);
}