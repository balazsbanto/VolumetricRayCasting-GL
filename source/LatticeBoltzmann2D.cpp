#include <LatticeBoltzmann2D.hpp>
#include <iomanip>

int fileIndex = 0;
// Start equilibrium distribution for the current initial config (D2Q9, rho = 10) 
const float F0_EQ = 4.4444444444f;
const float F1234_EQ = 1.1111111111f;
const float F5678_EQ = 0.2777777777f;

struct Distributions {
	float f0;
	cl::sycl::float4 f1234;
	cl::sycl::float4 f5678;
};

struct CellData
{
	Distributions distributions;
	cl::sycl::float2 velocity;
	bool cellType;

};

const auto colorFunction = [](cl::sycl::float2 inVelocity, bool isBoundary) {
	using namespace cl::sycl;
	float4 color = { 0.f, 0.f, 0.f, 1.f };

	// creat a color scale (use 4th value now for magnitude, later set the alpha channel here)
	float4 color1{ 0, 0, 0, 0.0 };
	float4 color2{ 0, 0, 1, 0.2 };
	float4 color3{ 0, 1, 1, 0.4 };
	float4 color4{ 0, 1, 0, 0.8 };
	float4 color5{ 1, 1, 0, 1.6 };
	float4 color6{ 1, 0, 0, 3.2 };

	if (isBoundary) {
		color = { 0.f, 0.f, 0.f, 1.f };
	}
	else {
		auto velocityMangitude = cl::sycl::length(inVelocity) * 20;

		int i = 0;
		float w;

		if (velocityMangitude <= color1.get_value(3))
		{
			color = color1;
		}
		else if (velocityMangitude >= color6.get_value(3))
		{
			color = color6;
		}
		else
		{
			float4 colorBoundaryStart;
			float4 colorBoundaryEnd;
			if ((float)color1.get_value(3) <= velocityMangitude && velocityMangitude < color2.get_value(3)) {
				colorBoundaryStart = color1;
				colorBoundaryEnd = color2;
			}
			else if ((float)color2.get_value(3) <= velocityMangitude && velocityMangitude < color3.get_value(3)) {
				colorBoundaryStart = color2;
				colorBoundaryEnd = color3;

			}
			else if ((float)color3.get_value(3) <= velocityMangitude && velocityMangitude < color4.get_value(3)) {
				colorBoundaryStart = color3;
				colorBoundaryEnd = color4;
			}
			else if ((float)color4.get_value(3) <= velocityMangitude && velocityMangitude < color5.get_value(3)) {
				colorBoundaryStart = color4;
				colorBoundaryEnd = color5;
			}
			else if ((float)color5.get_value(3) <= velocityMangitude && velocityMangitude < color6.get_value(3)) {
				colorBoundaryStart = color5;
				colorBoundaryEnd = color6;
			}

			// linear interpolation
			w = (velocityMangitude - colorBoundaryStart.get_value(3)) / (colorBoundaryEnd.get_value(3) - colorBoundaryStart.get_value(3));
			color = (1 - w) * colorBoundaryStart + w * colorBoundaryEnd;
		}
	}
	// set alpha to 1;
	color.set_value(3, 1.f);

	return color;
};


const auto computefEq = [](float rho, float weight, cl::sycl::float2 dir, cl::sycl::float2 velocity) {

	float u2 = cl::sycl::dot(velocity, velocity);
	float eu = cl::sycl::dot(dir, velocity);
	return rho * weight * (1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * u2);
};


const auto collide = [](const Distributions& cellDistributions, const bool cellType) {
	using namespace cl::sycl;
	//qDebug() << x << " " << y << " " << pos << "\n";
			// Read input distributions
	float f0 = cellDistributions.f0;
	float4 f1234 = cellDistributions.f1234;
	float4 f5678 = cellDistributions.f5678;
	bool type = cellType;

	const int dirX[9] = { 0, 1, 0, -1,  0, 1, -1,  -1,  1 };
	const int dirY[9] = { 0, 0, 1,  0, -1, 1,  1,  -1, -1 };

	const float weight[9] = { 4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0 };

	// (1/relaxation time) Related to viscosity 
	float omega = 1.2f;

	float rho;
	float2 u;

	//boundary
	if (type) {
		// Swap directions by swizzling // Ez igy nem jo, de ez csak a hataroknal jelent problemat
		/*f1234.x() = f1234.z();
		f1234.y() = f1234.w();
		f1234.z() = f1234.x();
		f1234.w() = f1234.y();

		f5678.x() = f5678.z();
		f5678.y() = f5678.w();
		f5678.z() = f5678.x();
		f5678.w() = f5678.y();*/

		rho = 0;
		u = float2{ 0.f, 0.f };
	}
	// fluid
	else
	{
		// Compute rho and u
		// Rho is computed by doing a reduction on f
		rho = f0 + f1234.get_value(0) + f1234.get_value(1) + f1234.get_value(2) + f1234.get_value(3)
			+ f5678.get_value(0) + f5678.get_value(1) + f5678.get_value(2) + f5678.get_value(3);

		// Compute velocity

		/*qDebug() << dirY[1] << " " << dirY[2] << " " << dirY[3] << " " << dirY[4] << " " << dirY[5] << " " << dirY[6] << " " << dirY[7] << " " << dirY[8] << "\n";
		qDebug() << dirX[1] << " " << dirX[2] << " " << dirX[3] << " " << dirX[4] << " " << dirX[5] << " " << dirX[6] << " " << dirX[7] << " " << dirX[8] << "\n";*/


		float uX = (f1234.get_value(0) * dirX[1] + f1234.get_value(1) * dirX[2] + f1234.get_value(2) * dirX[3] + f1234.get_value(3) * dirX[4]
			+ f5678.get_value(0) * dirX[5] + f5678.get_value(1) * dirX[6] + f5678.get_value(2) * dirX[7] + f5678.get_value(3) * dirX[8]) / rho;

		u.set_value(0, uX);

		float uY = (f1234.get_value(0) * dirY[1] + f1234.get_value(1) * dirY[2] + f1234.get_value(2) * dirY[3] + f1234.get_value(3) * dirY[4]
			+ f5678.get_value(0) * dirY[5] + f5678.get_value(1) * dirY[6] + f5678.get_value(2) * dirY[7] + f5678.get_value(3) * dirY[8]) / rho;

		u.set_value(1, uY);


		float4 fEq1234;	// Stores feq 
		float4 fEq5678;
		float fEq0;

		if (uX > 0 || uY > 0) {
			int breakpointHere = 1;
		}

		//Compute fEq
		fEq0 = computefEq(rho, weight[0], float2{ 0, 0 }, u);
		fEq1234.set_value(0, computefEq(rho, weight[1], float2{ dirX[1], dirY[1] }, u));
		fEq1234.set_value(1, computefEq(rho, weight[2], float2{ dirX[2], dirY[2] }, u));
		fEq1234.set_value(2, computefEq(rho, weight[3], float2{ dirX[3], dirY[3] }, u));
		fEq1234.set_value(3, computefEq(rho, weight[4], float2{ dirX[4], dirY[4] }, u));

		fEq5678.set_value(0, computefEq(rho, weight[5], float2{ dirX[5], dirY[5] }, u));
		fEq5678.set_value(1, computefEq(rho, weight[6], float2{ dirX[6], dirY[6] }, u));
		fEq5678.set_value(2, computefEq(rho, weight[7], float2{ dirX[7], dirY[7] }, u));
		fEq5678.set_value(3, computefEq(rho, weight[8], float2{ dirX[8], dirY[8] }, u));

		f0 = (1 - omega) * f0 + omega * fEq0;
		f1234 = (1 - omega) * f1234 + omega * fEq1234;
		f5678 = (1 - omega) * f5678 + omega * fEq5678;
	}

	return CellData{ Distributions{f0, f1234, f5678}, u, type };
};

template <cl::sycl::access::target Target>
const auto streamToNeighbours = [](const cl::sycl::int2 id, const int currentPos, const int width, const int height, const Distributions& currentCellDistributions,
	const cl::sycl::accessor<float, 1, cl::sycl::access::mode::discard_write, Target, cl::sycl::access::placeholder::false_t>& of0,
	const cl::sycl::accessor<cl::sycl::float4, 1, cl::sycl::access::mode::discard_write, Target, cl::sycl::access::placeholder::false_t>& of1234,
	const cl::sycl::accessor<cl::sycl::float4, 1, cl::sycl::access::mode::discard_write, Target, cl::sycl::access::placeholder::false_t>& of5678) {

	using namespace cl::sycl;
	// Propagate
	// New positions to write (Each thread will write 8 values)

	const int dirX[9] = { 0, 1, 0, -1,  0, 1, -1,  -1,  1 };
	const int dirY[9] = { 0, 0, 1,  0, -1, 1,  1,  -1, -1 };

	int8 x8 = int8(id.get_value(0));
	int8 y8 = int8(id.get_value(1));
	int8 width8 = int8(width);

	int8 nX = x8 + int8(dirX[1], dirX[2], dirX[3], dirX[4], dirX[5], dirX[6], dirX[7], dirX[8]);
	int8 nY = y8 + int8(dirY[1], dirY[2], dirY[3], dirY[4], dirY[5], dirY[6], dirY[7], dirY[8]);
	int8 nPos = nX + width8 * nY;


	int isNotRightBoundary = id.get_value(0) < int(width - 1); // Not on Right boundary
	int isNotUpperBoundary = id.get_value(1) > int(0);                      // Not on Upper boundary
	int isNotLeftBoundary = id.get_value(0) > int(0);                      // Not on Left boundary
	int isNotLowerBoundary = id.get_value(1) < int(height - 1); // Not on lower boundary

	of0[currentPos] = currentCellDistributions.f0;

	// Propagate to right cell
	if (isNotRightBoundary) {
		of1234[nPos.get_value(0)].set_value(0, currentCellDistributions.f1234.get_value(0));
	}

	// Propagate to Lower cell
	if (isNotLowerBoundary) {
		of1234[nPos.get_value(1)].set_value(1, currentCellDistributions.f1234.get_value(1));
	}

	// Propagate to left cell
	if (isNotLeftBoundary) {
		of1234[nPos.get_value(2)].set_value(2, currentCellDistributions.f1234.get_value(2));
	}

	// Propagate to Upper cell
	if (isNotUpperBoundary) {
		of1234[nPos.get_value(3)].set_value(3, currentCellDistributions.f1234.get_value(3));
	}

	// Propagate to Lower-Right cell
	if (isNotRightBoundary && isNotLowerBoundary) {
		of5678[nPos.get_value(4)].set_value(0, currentCellDistributions.f5678.get_value(0));
	}

	// Propogate to Lower-Left cell
	if (isNotLowerBoundary && isNotLeftBoundary) {
		of5678[nPos.get_value(5)].set_value(1, currentCellDistributions.f5678.get_value(1));
	}

	// Propagate to Upper-Left cell
	if (isNotLeftBoundary && isNotUpperBoundary) {
		of5678[nPos.get_value(6)].set_value(2, currentCellDistributions.f5678.get_value(2));
	}

	// Propagate to Upper-Right cell
	if (isNotUpperBoundary && isNotRightBoundary) {
		of5678[nPos.get_value(7)].set_value(3, currentCellDistributions.f5678.get_value(3));
	}

};

LatticeBoltzmann2DRaycaster::LatticeBoltzmann2DRaycaster(std::size_t plat,
    std::size_t dev,
    cl_bitfield type,
    QWindow* parent)
    : Raycaster(plat, dev, type, parent)
{
}

void LatticeBoltzmann2DRaycaster::mouseDragImpl(QMouseEvent* event_in) {
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

	auto if0 = f0_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read_write>();
	auto if1234 = f1234_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read_write>();
	auto if5678 = f5678_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read_write>();

	auto velocity_out = velocity_buffer->get_access<access::mode::read>();

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

void LatticeBoltzmann2DRaycaster::resetScene() {

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
}

void LatticeBoltzmann2DRaycaster::setInput() {
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

void LatticeBoltzmann2DRaycaster::writeOutputsToFile() {
	//fileIndex++;
	return;

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


	if (fileIndex == 3) {
		setInput();
		writeOutputsToFile();
	}
}

void LatticeBoltzmann2DRaycaster::runOnCPU() {
	using namespace cl::sycl;

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

	for (int y = 0; y < screen_height; y++) {
		for (int x = 0; x < screen_width; x++) {

			int2 id(x, y);
			int pos = id.get_value(0) + screen_width * id.get_value(1);

			auto cellAfterCollision = collide(Distributions{ if0[pos], if1234[pos], if5678[pos] }, type[pos]);

			streamToNeighbours<access::target::host_buffer>(id, pos, screen_width, screen_height, cellAfterCollision.distributions, of0, of1234, of5678);

			velocity_out[pos] = cellAfterCollision.velocity;

			auto finalPixelColor = colorFunction(cellAfterCollision.velocity, cellAfterCollision.cellType);


			/*if (finalPixelColor.get_value(0) > 0  || finalPixelColor.get_value(1) > 0 || finalPixelColor.get_value(2) > 0 )
				qDebug() << finalPixelColor.get_value(0) << " " << finalPixelColor.get_value(1) << " " << finalPixelColor.get_value(2) << finalPixelColor.get_value(3) << "\n";*/

		}
	}

	swapDataBuffers();

	writeOutputsToFile();

}

void LatticeBoltzmann2DRaycaster::updateSceneImpl() {
	using namespace cl::sycl;
	compute_queue.submit([&](cl::sycl::handler& cgh)
	{
		// Input buffers
		auto if0 = f0_buffers[Buffer::Front]->get_access<access::mode::read>(cgh);
		auto if1234 = f1234_buffers[Buffer::Front]->get_access<access::mode::read>(cgh);
		auto if5678 = f5678_buffers[Buffer::Front]->get_access<access::mode::read>(cgh);
		auto type = type_buffer.get_access<access::mode::read>(cgh);

		// Output buffers
		auto velocity_out = velocity_buffer->get_access<access::mode::discard_write>(cgh);
		const auto of0 = f0_buffers[Buffer::Back]->get_access<access::mode::discard_write>(cgh);
		const auto of1234 = f1234_buffers[Buffer::Back]->get_access<access::mode::discard_write>(cgh);
		const auto of5678 = f5678_buffers[Buffer::Back]->get_access<access::mode::discard_write>(cgh);

		auto new_lattice = latticeImages[Buffer::Back]->get_access<float4, access::mode::discard_write>(cgh);

		int screen_width = width();
		int screen_height = height();


		cgh.parallel_for<kernels::Lbm>(range<2>{ new_lattice.get_range() },
			[=, computefEq = computefEq, collide = collide, colorFunction = colorFunction, screen_width = screen_width, screen_height = screen_height](const item<2> i)
		{

			int2 id = (int2)i.get_id();

			int pos = id.get_value(0) + screen_width * id.get_value(1);

			auto cellAfterCollision = collide(Distributions{ if0[pos], if1234[pos], if5678[pos] }, type[pos]);

			streamToNeighbours<access::target::global_buffer>(id, pos, screen_width, screen_height, cellAfterCollision.distributions, of0, of1234, of5678);

			velocity_out[pos] = cellAfterCollision.velocity;
			auto finalPixelColor = colorFunction(cellAfterCollision.velocity, cellAfterCollision.cellType);

			auto setPixelForNewLattice = [=](float4 in) { new_lattice.write(id, in); };
			setPixelForNewLattice(finalPixelColor);

		});
	});
}


void LatticeBoltzmann2DRaycaster::swapDataBuffers() {
	std::swap(f0_buffers[Buffer::Front], f0_buffers[Buffer::Back]);
	std::swap(f1234_buffers[Buffer::Front], f1234_buffers[Buffer::Back]);
	std::swap(f5678_buffers[Buffer::Front], f5678_buffers[Buffer::Back]);
}