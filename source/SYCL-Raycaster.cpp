// Behavioral defines
//
// GLM
#define GLM_ENABLE_EXPERIMENTAL

#include <SYCL-Raycaster.hpp>
#include <glm/gtx/transform.hpp>

#include <iterator>
//#include "glm/ext.hpp"
//#include "C:/Diplomamunka/vcpkg/installed/x64-windows/include/glm/ext.hpp"

// Start equilibrium distribution for the current initial config (D2Q9, rho = 10) 
const float F0_EQ = 4.4444444f;
const float F1234_EQ = 1.111111f;
const float F5678_EQ = 0.277777f;

Raycaster::Raycaster(std::size_t plat,
               std::size_t dev,
               cl_bitfield type,
               QWindow *parent)
    : InteropWindow(plat, type, parent)
    , dev_id(dev)
    , cl_khr_gl_event_supported(false)
    , imageDrawn(false)
    , needMatrixReset(true)
	, phi(0)
	, theta(0)
{
}

// Override unimplemented InteropWindow function
void Raycaster::initializeGL()
{
    qDebug("Raycaster: Entering initializeGL");
    std::unique_ptr<QOpenGLDebugLogger> log(new QOpenGLDebugLogger(this));
    if (!log->initialize()) qWarning("Raycaster: QDebugLogger failed to initialize");

    // Initialize OpenGL resources
    vs = std::make_unique<QOpenGLShader>(QOpenGLShader::Vertex, this);
    fs = std::make_unique<QOpenGLShader>(QOpenGLShader::Fragment, this);
    sp = std::make_unique<QOpenGLShaderProgram>(this);
    vbo = std::make_unique<QOpenGLBuffer>(QOpenGLBuffer::VertexBuffer);
    vao = std::make_unique<QOpenGLVertexArrayObject>(this);
    texs = { std::make_unique<QOpenGLTexture>(QOpenGLTexture::Target::Target2D),
             std::make_unique<QOpenGLTexture>(QOpenGLTexture::Target::Target2D) };

    // Initialize frame buffer
    glFuncs->glViewport(0, 0, width(), height());
    glFuncs->glClearColor(0.0, 0.0, 0.0, 1.0);
    glFuncs->glDisable(GL_DEPTH_TEST);
    glFuncs->glDisable(GL_CULL_FACE);

    // Create shaders
    qDebug("Raycaster: Building shaders...");
    if (!vs->compileSourceFile( (shader_location + "/Vertex.glsl").c_str())) qWarning("%s", vs->log().data());
    if (!fs->compileSourceFile( (shader_location + "/Fragment.glsl").c_str())) qWarning("%s", fs->log().data());
    qDebug("Raycaster: Done building shaders");

    // Create and link shaderprogram
    qDebug("Raycaster: Linking shaders...");
    if (!sp->addShader(vs.get())) qWarning("Raycaster: Could not add vertex shader to shader program");
    if (!sp->addShader(fs.get())) qWarning("Raycaster: Could not add fragment shader to shader program");
    if (!sp->link()) qWarning("%s", sp->log().data());
    qDebug("Raycaster: Done linking shaders");

    // Init device memory
    qDebug("Raycaster: Initializing OpenGL buffers...");

    std::vector<float> quad =
        //  vertices  , tex coords
        //  x  ,   y  ,  u  ,   v
        { -1.0f, -1.0f, 0.0f, 0.0f,
          -1.0f,  1.0f, 0.0f, 1.0f,
           1.0f, -1.0f, 1.0f, 0.0f,
           1.0f,  1.0f, 1.0f, 1.0f };

    if (!vbo->create()) qWarning("Raycaster: Could not create VBO");
    if (!vbo->bind()) qWarning("Raycaster: Could not bind VBO");
    vbo->setUsagePattern(QOpenGLBuffer::StaticDraw);
    vbo->allocate(quad.data(), (int)quad.size() * sizeof(float));
    vbo->release();

    qDebug("Raycaster: Done initializing OpenGL buffers");

    // Setup VAO for the VBO
    if (!vao->create()) qWarning("Raycaster: Could not create VAO");

    vao->bind();
    {
        if (!vbo->bind()) qWarning("Raycaster: Could not bind VBO");

        // Setup shader attributes (can only be done when a VBO is bound, VAO does not store shader state
        if (!sp->bind()) qWarning("Raycaster: Failed to bind shaderprogram");
        sp->enableAttributeArray(0);
        sp->enableAttributeArray(1);
        sp->setAttributeArray(0, GL_FLOAT, (GLvoid *)(NULL), 2, sizeof(cl::sycl::float4));
        sp->setAttributeArray(1, GL_FLOAT, (GLvoid *)(NULL + 2 * sizeof(float)), 2, sizeof(cl::sycl::float4));
        sp->release();
    }
    vao->release();

    std::vector<std::array<float, 4>> texels;
    std::generate_n(std::back_inserter(texels),
                    width() * height(),
                    [prng = std::default_random_engine{},
                     dist = std::uniform_int_distribution<std::uint32_t>{ 0, 0 }]() mutable
    {
        auto rand = dist(prng);
        return std::array<float, 4>{ (float)rand, (float)rand, (float)rand, 0.f };
    });

    // Quote from the QOpenGLTexture documentation of Qt 5.12
    //
    // The typical usage pattern for QOpenGLTexture is:
    //  -  Instantiate the object specifying the texture target type
    //  -  Set properties that affect the storage requirements e.g.storage format, dimensions
    //  -  Allocate the server - side storage
    //  -  Optionally upload pixel data
    //  -  Optionally set any additional properties e.g.filtering and border options
    //  -  Render with texture or render to texture

    for (auto& tex : texs)
    {
        tex->setSize(width(), height());
        tex->setFormat(QOpenGLTexture::TextureFormat::RGBA32F);
        tex->allocateStorage(QOpenGLTexture::PixelFormat::RGBA, QOpenGLTexture::PixelType::Float32);
        tex->setData(QOpenGLTexture::PixelFormat::RGBA, QOpenGLTexture::PixelType::Float32, texels.data());
        tex->generateMipMaps();
    }

    for (const QOpenGLDebugMessage& message : log->loggedMessages()) qDebug() << message << "\n";

	setMatrices();
	resetLBM();

     qDebug("Raycaster: Leaving initializeGL");
}

// Override unimplemented InteropWindow function
void Raycaster::initializeCL()
{
    qDebug("Raycaster: Entering initializeCL");

    // Translate OpenGL handles to OpenCL
    std::transform(texs.cbegin(), texs.cend(), CL_latticeImages.begin(), [this](const std::unique_ptr<QOpenGLTexture>& tex)
    {
        return cl::ImageGL{ CLcontext(), CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, tex->textureId() };
    });

    // Translate OpenCL handles to SYCL
    auto async_error_handler = [](cl::sycl::exception_list errors)
    {
        for (auto error : errors)
        {
            try { std::rethrow_exception(error); }
            catch (cl::sycl::exception e)
            {
                qDebug() << e.what();
                std::exit(e.get_cl_code());
            }
            catch (std::exception e)
            {
                qDebug() << e.what();
                std::exit(EXIT_FAILURE);
            }
        }
    };

    try
    {
        context = cl::sycl::context{ CLcontext()(), async_error_handler };
        device = cl::sycl::device{ CLdevices().at(dev_id)() };
        compute_queue = cl::sycl::queue{ CLcommandqueues().at(dev_id)(), context };

        std::transform(CL_latticeImages.cbegin(), CL_latticeImages.cend(), latticeImages.begin(), [this](const cl::ImageGL & image)
        {
            return std::make_unique<cl::sycl::image<2>>(image(), context);
        });
    }
    catch(cl::sycl::exception e)
    {
        qDebug() << e.what();
        std::exit(e.get_cl_code());
    }

    qDebug("Raycaster: Querying device capabilities");
    auto extensions = device.get_info<cl::sycl::info::device::extensions>();
    cl_khr_gl_event_supported = std::find(extensions.cbegin(), extensions.cend(), "cl_khr_gl_event") != extensions.cend();

    // Init bloat vars
    std::copy(CL_latticeImages.cbegin(), CL_latticeImages.cend(), std::back_inserter(interop_resources));

    qDebug("Raycaster: Leaving initializeCL");
}

// Override unimplemented InteropWindow function
void Raycaster::render()
{
    std::unique_ptr<QOpenGLDebugLogger> log(new QOpenGLDebugLogger(this));
    if (!log->initialize()) qWarning("Raycaster: QDebugLogger failed to initialize");

    // Update matrices as needed
    if(needMatrixReset) setVRMatrices();

    // Clear Frame Buffer and Z-Buffer
    glFuncs->glClear(GL_COLOR_BUFFER_BIT);

    // Draw
    if(!sp->bind()) qWarning("QGripper: Failed to bind shaderprogram");
    vao->bind();

    texs[Buffer::Front]->bind();

    glFuncs->glDrawArrays(GL_TRIANGLE_STRIP, 0, static_cast<GLsizei>(4));

    texs[Buffer::Front]->release();
    vao->release();
    sp->release();

    // Wait for all drawing commands to finish
    if (!cl_khr_gl_event_supported) glFuncs->glFinish();
    else glFuncs->glFlush();

    imageDrawn = true;

    for (const QOpenGLDebugMessage& message : log->loggedMessages()) qDebug() << message << "\n";
}

// Override unimplemented InteropWindow function
void Raycaster::render(QPainter* painter)
{
    QString text("QGripper: ");
    text.append("IPS = ");
    text.append(QString::number(getActIPS()));
    text.append(" | FPS = ");
    text.append(QString::number(getActFPS()));
    
    this->setTitle(text);
}

// Override InteropWindow function
void Raycaster::resizeGL(QResizeEvent* event_in)
{
    glFuncs->glViewport(0, 0, event_in->size().width(), event_in->size().height());
    checkGLerror();

    needMatrixReset = true; // projection matrix need to be recalculated
}

// Override InteropWindow function
bool Raycaster::event(QEvent *event_in)
{
	QMouseEvent* mouse_event;
	QWheelEvent* wheel_event;
    QKeyEvent* keyboard_event;

    // Process messages arriving from application
    switch (event_in->type())
    {
	case QEvent::MouseMove:
		mouse_event = static_cast<QMouseEvent*>(event_in);

		if ((mouse_event->buttons() & Qt::MouseButton::LeftButton) && // If RMB is pressed AND
			(mousePos != mouse_event->pos()))                          // Mouse has moved 
			mouseDrag(mouse_event);

		mousePos = mouse_event->pos();
		return true;
	
	case QEvent::KeyPress:
        keyboard_event = static_cast<QKeyEvent*>(event_in);

        if(keyboard_event->key() == Qt::Key::Key_Space) setAnimating(!getAnimating());
        return true;

    default:
        // In case InteropWindow does not implement handling of the even, we pass it on to the base class
        return InteropWindow::event(event_in);
    }
}

// setCameraMatrices, ezeket csak a sycl-ben hasznalni. Ezt kellene meghivni akkor, amikor kapok egy uj eventet.

// Input handler function
void Raycaster::mouseDrag(QMouseEvent* event_in)
{

	using namespace cl::sycl;
	/*qDebug() << "evenX " << event_in->x() << "oldX " << mousePos.x() << "\n";
	qDebug() << "evenY " << event_in->y() << "oldY " << mousePos.y() << "\n";*/
	// TODO: check how the sign in the y direction works
	phi = (event_in->x() - mousePos.x());
	theta = (event_in->y() - mousePos.y());

	// check here which kernel we are running. Run this only in case of LBM
	if (true) {
		float2 dragVelocity{ event_in->x() - mousePos.x(),  mousePos.y() - event_in->y() };
		auto magnitude = length(dragVelocity);
		dragVelocity /= (1 + 2 * magnitude);

		// Set new distributions
		float rho;

		int x = event_in->x();
		int y = height() - 1 - event_in->y();
		int pos = x + width() * y;

		auto if0 = f0_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read_write>();
		auto if1234 = f1234_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read_write>();
		auto if5678 = f5678_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read_write>();

		auto velocity_out = velocity_buffer.get_access<access::mode::read>();

		// Calculate density from input distribution
		rho = if0[pos]
			+ if1234[pos].get_value(0) + if1234[pos].get_value(1) + if1234[pos].get_value(2) + if1234[pos].get_value(3) +
			+if5678[pos].get_value(0) + if5678[pos].get_value(1) + if5678[pos].get_value(2) + if5678[pos].get_value(3);

		// Increase the speed by input speed
		//velocity_out[pos] += dragVelocity;

		float2 newVel = velocity_out[pos] + dragVelocity;

		// Calculate new distribution based on input speed

		if0[pos] = computefEq(w[0], cl::sycl::float2{ h_dirX[0], h_dirY[0] }, rho, newVel);

		if1234[pos].set_value(0, computefEq(w[1], cl::sycl::float2{ h_dirX[1], h_dirY[1] }, rho, newVel));
		if1234[pos].set_value(1, computefEq(w[2], cl::sycl::float2{ h_dirX[2], h_dirY[2] }, rho, newVel));
		if1234[pos].set_value(2, computefEq(w[3], cl::sycl::float2{ h_dirX[3], h_dirY[3] }, rho, newVel));
		if1234[pos].set_value(3, computefEq(w[4], cl::sycl::float2{ h_dirX[4], h_dirY[4] }, rho, newVel));

		if5678[pos].set_value(0, computefEq(w[5], cl::sycl::float2{ h_dirX[5], h_dirY[5] }, rho, newVel));
		if5678[pos].set_value(1, computefEq(w[6], cl::sycl::float2{ h_dirX[6], h_dirY[6] }, rho, newVel));
		if5678[pos].set_value(2, computefEq(w[7], cl::sycl::float2{ h_dirX[7], h_dirY[7] }, rho, newVel));
		if5678[pos].set_value(3, computefEq(w[8], cl::sycl::float2{ h_dirX[8], h_dirY[8] }, rho, newVel));

	}
	/*qDebug() << "event_in: " << event_in->x() << " " << event_in->y();
	qDebug() << "mousePos: " << mousePos.x() << " " <<  mousePos.y() ;
	qDebug() << "phi: " << phi ;
	qDebug() << "theta: " << theta;*/
	
	needMatrixReset = true;

	if (!getAnimating()) renderNow();
}



void Raycaster::setVRMatrices() {

	glm::mat3 tansformation = glm::rotate(glm::radians(phi), glm::vec3(0.f, 1.f, 0.f)) *
		glm::rotate(glm::radians(theta), glm::vec3(1.f, 0.f, 0.f));
	m_vecEye -= m_vecTarget;
	m_vecEye = tansformation * m_vecEye;
	m_vecEye += m_vecTarget;

	m_vecUp = tansformation * m_vecUp;
	m_vecUp = glm::normalize(m_vecUp);

	glm::mat4 worldToView = glm::lookAt(m_vecEye, m_vecTarget, m_vecUp);
	m_viewToWorldMtx = glm::inverse(worldToView);

	needMatrixReset = false;
}
// Helper function
void Raycaster::setMatrices()
{
    // Set camera to view the origo from the z-axis with up along the y-axis
    // and distance so the entire sim space is visible with given field-of-view
    QVector3D vecTarget{ 0, 0, 0 };
    QVector3D vecUp{ 0, 1, 0 };
	QVector3D vecEye = vecTarget + QVector3D{ 0, 0, 1 };

    QMatrix4x4 matWorld; // Identity

    QMatrix4x4 matView; // Identity
	matView.lookAt(vecEye, vecTarget, vecUp);

    QMatrix4x4 matProj; // Identity

	matProj.ortho((float)width(), (float)width(),
				  (float)height(), (float)height(),
                  std::numeric_limits<float>::epsilon(),  
                  std::numeric_limits<float>::max());


	//qDebug() << matProj << "\n"; 

    sp->bind();
    sp->setUniformValue("mat_MVP", matProj * matView * matWorld);
    sp->release();

    needMatrixReset = false;

	// Set glm matrix for use inside the SYCL kerlen, because QVector may be not be supported inside SDY 
	m_vecTarget = glm::vec3(0.f, 0.f, 0.f);
	m_vecUp = glm::vec3(0.f, 1.f, 0.f);
	m_vecEye = m_vecTarget + glm::vec3(0, 0, 2.2f);
	glm::mat4 worldToView = glm::lookAt(m_vecEye, m_vecTarget, m_vecUp);

	//qDebug() << glm::to_string(matWorld).c_str() << "\n";
	//matWorld = glm::rotate(glm::radians(theta), glm::vec3( 1, 0, 0));
	//matWorld = glm::rotate(glm::radians(phi), glm::vec3( 0, 0, 1));
	//matWorld = glm::rotate(matWorld, glm::radians(phi), glm::vec3(0, 0, 0.1f));

	m_viewToWorldMtx = glm::inverse(worldToView);
}


size_t Raycaster::getMeshSize() {
	return width() * height();
}


void Raycaster::resetLBM() {
	using namespace cl::sycl;
	
	// Initial velocity is 0
	type_host =  new bool[getMeshSize()];
	f0_host [Buffer::Front]   = std::vector<float>(getMeshSize(), F0_EQ );
	f1234_host[Buffer::Front] = std::vector<float4>(getMeshSize(), float4{ F1234_EQ });
	f5678_host[Buffer::Front] = std::vector<float4>(getMeshSize(), float4{ F5678_EQ });

	f0_host[Buffer::Back]	 = f0_host[Buffer::Front];
	f1234_host[Buffer::Back] = f1234_host[Buffer::Front];
	f5678_host[Buffer::Back] = f5678_host[Buffer::Front];

	f0_buffers[Buffer::Front] = std::make_unique<buffer<float, 1>>(f0_host[Buffer::Front].data(), range<1> {getMeshSize()});
	f1234_buffers[Buffer::Front] = std::make_unique<buffer<float4, 1>>(f1234_host[Buffer::Front].data(), range<1> {getMeshSize()});
	f5678_buffers[Buffer::Front] = std::make_unique<buffer<float4, 1>>(f5678_host[Buffer::Front].data(), range<1> { getMeshSize()});
	type_buffer = buffer<bool, 1>{ type_host, range<1> {getMeshSize()} };


	f0_buffers[Buffer::Back] = std::make_unique<buffer<float, 1>>(f0_host[Buffer::Back].data(), range<1> {getMeshSize()});
	f1234_buffers[Buffer::Back] = std::make_unique<buffer<float4, 1>>(f1234_host[Buffer::Back].data(), range<1> {getMeshSize()});
	f5678_buffers[Buffer::Back] = std::make_unique<buffer<float4, 1>>(f5678_host[Buffer::Back].data(), range<1> { getMeshSize()});

	velocity_host = std::vector<float2>(getMeshSize(), float2{ 0 });
	velocity_buffer = buffer<float2, 1>{ velocity_host.data(), range<1> { getMeshSize()} };

	// Vector with contants
	h_dirX_buffer = buffer<int, 1>{ h_dirX.data(), range<1> {h_dirX.size()} };
	h_dirY_buffer = buffer<int, 1>{ h_dirY.data(), range<1> {h_dirY.size()} };
	h_weigt_buffer = buffer<float, 1>{ w.data(), range<1> {w.size()} };
	


	//float2 u0 = { 0.f, 0.f };

	for (int y = 0; y < height(); y++) {
		for (int x = 0; x < width(); x++) {

			int pos = x + y * width();

			// Initialize boundary cells
			//if (x == 0 || x == (width() - 1) || y == 0 || y == (height() - 1))
			//{
			//	type[pos] = 1;
			//}

			//// Initialize fluid cells
			//else
			//{
			type_host[pos] = 0;
			//}

		}
	}

	// Set new distributions
	//float rho;

	//int x = 1;
	//int y = height() - 1 - 1;
	//int pos = x + width() * y;

	//auto if0 = f0_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read_write>();
	//auto if1234 = f1234_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read_write>();
	//auto if5678 = f5678_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read_write>();

	//auto dirX = h_dirX_buffer.get_access<access::mode::read>();
	//auto dirY = h_dirY_buffer.get_access<access::mode::read>();
	//auto weight = h_weigt_buffer.get_access<access::mode::read>();


	//// Calculate density from input distribution
	//rho = if0[pos]
	//	+ if1234[pos].get_value(0) + if1234[pos].get_value(1) + if1234[pos].get_value(2) + if1234[pos].get_value(3) +
	//	+ if5678[pos].get_value(0) + if5678[pos].get_value(1) + if5678[pos].get_value(2) + if5678[pos].get_value(3);

	//float2 newVel = float2{ 0.34, -0.34 };

	//// Calculate new distribution based on input speed
	//if0[pos] = computefEq(weight[0], cl::sycl::float2{ dirX[0], dirY[0] }, rho, newVel);

	//if1234[pos].set_value(0, computefEq(weight[1], cl::sycl::float2{ dirX[1], dirY[1] }, rho, newVel));
	//if1234[pos].set_value(1, computefEq(weight[2], cl::sycl::float2{ dirX[2], dirY[2] }, rho, newVel));
	//if1234[pos].set_value(2, computefEq(weight[3], cl::sycl::float2{ dirX[3], dirY[3] }, rho, newVel));
	//if1234[pos].set_value(3, computefEq(weight[4], cl::sycl::float2{ dirX[4], dirY[4] }, rho, newVel));
	//							 
	//if5678[pos].set_value(0, computefEq(weight[5], cl::sycl::float2{ dirX[5], dirY[5] }, rho, newVel));
	//if5678[pos].set_value(1, computefEq(weight[6], cl::sycl::float2{ dirX[6], dirY[6] }, rho, newVel));
	//if5678[pos].set_value(2, computefEq(weight[7], cl::sycl::float2{ dirX[7], dirY[7] }, rho, newVel));
	//if5678[pos].set_value(3, computefEq(weight[8], cl::sycl::float2{ dirX[8], dirY[8] }, rho, newVel));
	//
	//testOutputs();
}


float Raycaster::computefEq(float weight, cl::sycl::float2 dir, float rho, cl::sycl::float2 velocity) {

	float u2 = cl::sycl::dot(velocity, velocity);
	float eu = cl::sycl::dot(dir, velocity);  
	return rho * weight * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2);
}

void Raycaster::runOnCPU() {
	using namespace cl::sycl;

	auto if0 = f0_buffers[Buffer::Front]->get_access<access::mode::read>();
	auto if1234 = f1234_buffers[Buffer::Front]->get_access<access::mode::read>();
	auto if5678 = f5678_buffers[Buffer::Front]->get_access<access::mode::read>();
	auto type = type_buffer.get_access<access::mode::read>();

	// Output
	auto of0 = f0_buffers[Buffer::Back]->get_access<access::mode::write>();
	auto of1234 = f1234_buffers[Buffer::Back]->get_access<access::mode::write>();
	auto of5678 = f5678_buffers[Buffer::Back]->get_access<access::mode::write>();
	auto velocity_out = velocity_buffer.get_access<access::mode::write>();;

	// Vector with contants
	auto dirX = h_dirX_buffer.get_access<access::mode::read>();
	auto dirY = h_dirY_buffer.get_access<access::mode::read>();
	auto weight = h_weigt_buffer.get_access<access::mode::read>();

	auto om = omega;

	for (int y = 0; y < height(); y++) {
		for (int x = 0; x < width(); x++) {

			int2 id(x, y);
			int pos = id.get_value(0) + width() * id.get_value(1);

			qDebug() << x << " " << y << " " << pos << "\n";
			// Read input distributions
			float f0 = if0[pos];
			float4 f1234 = if1234[pos];
			float4 f5678 = if5678[pos];

			float rho;	//Density
			float2 u;	//Velocity

			// Collide
			//boundary
			if (type[pos]) {
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

				// TODO: check if I use f1234.get_value(index) instead
				float uX = (f1234.get_value(0) * float(dirX[1]) + f1234.get_value(1) * float(dirX[2]) + f1234.get_value(2) * float(dirX[3]) + f1234.get_value(3) * float(dirX[4])
					+ f5678.get_value(0) * float(dirX[5]) + f5678.get_value(1) * float(dirX[6]) + f5678.get_value(2) * float(dirX[7]) + f5678.get_value(3) * float(dirX[8])) / rho;

				u.set_value(0, uX);

				float uY = (f1234.get_value(0) * float(dirY[1]) + f1234.get_value(1) * float(dirY[2]) + f1234.get_value(2) * float(dirY[3]) + f1234.get_value(3) * float(dirY[4])
					+ f5678.get_value(0) * float(dirY[5]) + f5678.get_value(1) * float(dirY[6]) + f5678.get_value(2) * float(dirY[7]) + f5678.get_value(3) * float(dirY[8])) / rho;

				u.set_value(1, uY);

				float4 fEq1234;	// Stores feq 
				float4 fEq5678;
				float fEq0;

				auto computefEq = [](float rho, float weight, float2 dir, float2 u) {
					float u2 = dot(u, u);
					float eu = dot(dir, u);
					return rho * weight * (1.0f + (3.0f * eu) + (4.5f * eu * eu) - (1.5f * u2));

				};

				// Compute fEq
				fEq0 = computefEq(rho, weight[0], float2{ 0, 0 }, u);
				fEq1234.set_value(0, computefEq(rho, weight[1], float2{ dirX[1], dirY[1] }, u));
				fEq1234.set_value(1, computefEq(rho, weight[2], float2{ dirX[2], dirY[2] }, u));
				fEq1234.set_value(2, computefEq(rho, weight[3], float2{ dirX[3], dirY[3] }, u));
				fEq1234.set_value(3, computefEq(rho, weight[4], float2{ dirX[4], dirY[4] }, u));

				fEq5678.set_value(0, computefEq(rho, weight[5], float2{ dirX[5], dirY[5] }, u));
				fEq5678.set_value(1, computefEq(rho, weight[6], float2{ dirX[6], dirY[6] }, u));
				fEq5678.set_value(2, computefEq(rho, weight[7], float2{ dirX[7], dirY[7] }, u));
				fEq5678.set_value(3, computefEq(rho, weight[8], float2{ dirX[8], dirY[8] }, u));

				f0 = (1 - om) * f0 + om * fEq0;
				f1234 = (1 - om) * f1234 + om * fEq1234;
				f5678 = (1 - om) * f5678 + om * fEq5678;
			}

			velocity_out[pos] = u;

			// Propagate
			// New positions to write (Each thread will write 8 values)

			int8 x8 = int8(id.get_value(0));
			int8 y8 = int8(id.get_value(1));
			int8 width8 = int8(width());

			int8 nX = x8 + int8(dirX[1], dirX[2], dirX[3], dirX[4], dirX[5], dirX[6], dirX[7], dirX[8]);
			int8 nY = y8 + int8(dirY[1], dirY[2], dirY[3], dirY[4], dirY[5], dirY[6], dirY[7], dirY[8]);
			int8 nPos = nX + width8 * nY;

			// Write center distribution to thread's location
			of0[pos] = f0;

			int t1 = id.get_value(0) < int(width() - 1); // Not on Right boundary
			int t4 = id.get_value(1) > int(0);                      // Not on Upper boundary
			int t3 = id.get_value(0) > int(0);                      // Not on Left boundary
			int t2 = id.get_value(1) < int(height() - 1); // Not on lower boundary

			// Propagate to right cell
			if (t1) {
				of1234[nPos.get_value(0)].set_value(0, f1234.get_value(0));
				//qDebug() << "t1" << of1234[nPos.get_value(0)].get_value(0) << "\n";
			}

			// Propagate to Lower cell
			if (t2) {
				of1234[nPos.get_value(1)].set_value(1, f1234.get_value(1));
				//qDebug() << "t2" << of1234[nPos.get_value(1)].get_value(1) << "\n";
			}

			// Propagate to left cell
			if (t3) {
				of1234[nPos.get_value(2)].set_value(2, f1234.get_value(2));
				//qDebug() << "t3" << of1234[nPos.get_value(2)].get_value(2) << "\n";
			}

			// Propagate to Upper cell
			if (t4) {
				of1234[nPos.get_value(3)].set_value(3, f1234.get_value(3));
				//qDebug() << "t4" << of1234[nPos.get_value(3)].get_value(3) << "\n";
			}

			// Propagate to Lower-Right cell
			if (t1 && t2) {
				of5678[nPos.get_value(4)].set_value(0, f5678.get_value(0));
				//qDebug() << "t5" << nPos.get_value(4) << "\n";
				//qDebug() << "t5" << of5678[nPos.get_value(4)].get_value(0) << "\n";
			}

			// Propogate to Lower-Left cell
			if (t2 && t3) {
				of5678[nPos.get_value(5)].set_value(1, f5678.get_value(1));
				//qDebug() << "t6" << of5678[nPos.get_value(5)].get_value(1) << "\n";
			}

			// Propagate to Upper-Left cell
			if (t3 && t4) {
				of5678[nPos.get_value(6)].set_value(2, f5678.get_value(2));
				//qDebug() << "t7" << of5678[nPos.get_value(6)].get_value(2) << "\n";
			}

			// Propagate to Upper-Right cell
			if (t4 && t1) {
				of5678[nPos.get_value(7)].set_value(3, f5678.get_value(3));
				//qDebug() << "t8" << of5678[nPos.get_value(7)].get_value(3) << "\n";
			}

			auto getColor = [](float2 inVelocity, bool isBoundary) {
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

					if (velocityMangitude < color1.get_value(3))
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

				return color * 255;
			};

			auto finalPixelColor = getColor(u, type[pos]);

		}
	}

	std::swap(f0_buffers[Buffer::Front], f0_buffers[Buffer::Back]);
	std::swap(f1234_buffers[Buffer::Front], f1234_buffers[Buffer::Back]);
	std::swap(f5678_buffers[Buffer::Front], f5678_buffers[Buffer::Back]);
	testOutputs();



	/*std::ofstream of0_file("of1234.txt");
	std::ostream_iterator<float4> output_it(of0_file, " ");
	std::copy(of1234.begin(), of1234.end(), output_it);
	of0_file.close();*/
}

// LBM
void Raycaster::updateScene()
{
	// NOTE 1: When cl_khr_gl_event is NOT supported, then clFinish() is the only portable
	//         sync method and hence that will be called.
	//
	// NOTE 2.1: When cl_khr_gl_event IS supported AND the possibly conflicting OpenGL
	//           context is current to the thread, then it is sufficient to wait for events
	//           of clEnqueueAcquireGLObjects, as the spec guarantees that all OpenGL
	//           operations involving the acquired memory objects have finished. It also
	//           guarantees that any OpenGL commands issued after clEnqueueReleaseGLObjects
	//           will not execute until the release is complete.
	//         
	//           See: opencl-1.2-extensions.pdf (Rev. 15. Chapter 9.8.5)

	//runOnCPU();
	//return;

	cl::Event acquire, release;
	
	CLcommandqueues().at(dev_id).enqueueAcquireGLObjects(&interop_resources, nullptr, &acquire);
	using namespace cl::sycl;

	try
	{
		compute_queue.submit([&](cl::sycl::handler& cgh)
		{
			// Input buffers
			auto if0 = f0_buffers[Buffer::Front]->get_access<access::mode::read>(cgh);
			auto if1234 = f1234_buffers[Buffer::Front]->get_access<access::mode::read>(cgh);
			auto if5678 = f5678_buffers[Buffer::Front]->get_access<access::mode::read>(cgh);
			auto type = type_buffer.get_access<access::mode::read>(cgh);
			
			// Output buffers
			auto velocity_out = velocity_buffer.get_access<access::mode::discard_write>(cgh);
			auto of0 = f0_buffers[Buffer::Back]->get_access<access::mode::discard_write>(cgh);
			auto of1234 = f1234_buffers[Buffer::Back]->get_access<access::mode::discard_write>(cgh);
			auto of5678 = f5678_buffers[Buffer::Back]->get_access<access::mode::discard_write>(cgh);

			// Vector with contants
			auto dirX = h_dirX_buffer.get_access<access::mode::read>(cgh);
			auto dirY = h_dirY_buffer.get_access<access::mode::read>(cgh);
			auto weight = h_weigt_buffer.get_access<access::mode::read>(cgh);
			//auto colorScale = h_colorScale_mrgb_buffer.get_access<access::mode::read>(cgh);

			auto old_lattice = latticeImages[Buffer::Front]->get_access<float4, access::mode::read>(cgh);
			auto new_lattice = latticeImages[Buffer::Back]->get_access<float4, access::mode::write>(cgh);

			sampler periodic{ coordinate_normalization_mode::unnormalized,
							  addressing_mode::none,
							  filtering_mode::nearest };

			int screen_width = width();
			int screen_height = height();

			cgh.parallel_for<kernels::Lbm>(range<2>{ old_lattice.get_range() },
				[=, om = omega, width = screen_width, height = screen_height](const item<2> i)
			{

				//auto getPixelFromOldLattice = [=](int2 in) { return old_lattice.read(in, periodic); };
				auto setPixelForNewLattice = [=](float4 in) { new_lattice.write((int2)i.get_id(), in); };

				int2 id = (int2)i.get_id();
				int pos = id.get_value(0) + width * id.get_value(1);

				// Read input distributions
				float f0 = if0[pos];
				float4 f1234 = if1234[pos];
				float4 f5678 = if5678[pos];

				float rho;	//Density
				float2 u;	//Velocity

				// Collide
				//boundary
				if (type[pos]) {
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

					// TODO: check if I use f1234.get_value(index) instead
					float uX = (f1234.get_value(0) * float(dirX[1]) + f1234.get_value(1) * float(dirX[2]) + f1234.get_value(2) * float(dirX[3]) + f1234.get_value(3) * float(dirX[4])
						+ f5678.get_value(0) * float(dirX[5]) + f5678.get_value(1) * float(dirX[6]) + f5678.get_value(2) * float(dirX[7]) + f5678.get_value(3) * float(dirX[8])) / rho;

					u.set_value(0, uX);

					float uY = (f1234.get_value(0) * float(dirY[1]) + f1234.get_value(1) * float(dirY[2]) + f1234.get_value(2) * float(dirY[3]) + f1234.get_value(3) * float(dirY[4])
						+ f5678.get_value(0) * float(dirY[5]) + f5678.get_value(1) * float(dirY[6]) + f5678.get_value(2) * float(dirY[7]) + f5678.get_value(3) * float(dirY[8])) / rho;

					u.set_value(1, uY);

					float4 fEq1234;	// Stores feq 
					float4 fEq5678;
					float fEq0;

					auto computefEq = [](float rho, float weight, float2 dir, float2 u) {
						float u2 = dot(u, u);
						float eu = dot(dir, u);
						return rho * weight * (1.0f + (3.0f * eu) + (4.5f * eu * eu) - (1.5f * u2));

					};

					// Compute fEq
					fEq0 = computefEq(rho, weight[0], float2{ 0, 0 }, u);
					fEq1234.set_value(0, computefEq(rho, weight[1], float2{ dirX[1], dirY[1] }, u));
					fEq1234.set_value(1, computefEq(rho, weight[2], float2{ dirX[2], dirY[2] }, u));
					fEq1234.set_value(2, computefEq(rho, weight[3], float2{ dirX[3], dirY[3] }, u));
					fEq1234.set_value(3, computefEq(rho, weight[4], float2{ dirX[4], dirY[4] }, u));

					fEq5678.set_value(0, computefEq(rho, weight[5], float2{ dirX[5], dirY[5] }, u));
					fEq5678.set_value(1, computefEq(rho, weight[6], float2{ dirX[6], dirY[6] }, u));
					fEq5678.set_value(2, computefEq(rho, weight[7], float2{ dirX[7], dirY[7] }, u));
					fEq5678.set_value(3, computefEq(rho, weight[8], float2{ dirX[8], dirY[8] }, u));

					f0 = (1 - om) * f0 + om * fEq0;
					f1234 = (1 - om) * f1234 + om * fEq1234;
					f5678 = (1 - om) * f5678 + om * fEq5678;
				}

				velocity_out[pos] = u;

				// Propagate
				// New positions to write (Each thread will write 8 values)

				int8 x8 = int8(id.get_value(0));
				int8 y8 = int8(id.get_value(1));
				int8 width8 = int8(width);

				int8 nX = x8 + int8(dirX[1], dirX[2], dirX[3], dirX[4], dirX[5], dirX[6], dirX[7], dirX[8]);
				int8 nY = y8 + int8(dirY[1], dirY[2], dirY[3], dirY[4], dirY[5], dirY[6], dirY[7], dirY[8]);
				int8 nPos = nX + width8 * nY;

				// Write center distribution to thread's location
				of0[pos] = f0;

				int t1 = id.get_value(0) < int(width - 1); // Not on Right boundary
				int t4 = id.get_value(1) > int(0);                      // Not on Upper boundary
				int t3 = id.get_value(0) > int(0);                      // Not on Left boundary
				int t2 = id.get_value(1) < int(height - 1); // Not on lower boundary

				// Propagate to right cell
				if (t1) {
					of1234[nPos.get_value(0)].set_value(0, f1234.get_value(0));
				}

				// Propagate to Lower cell
				if (t2) {
					of1234[nPos.get_value(1)].set_value(1, f1234.get_value(1));
				}

				// Propagate to left cell
				if (t3) {
					of1234[nPos.get_value(2)].set_value(2, f1234.get_value(2));
				}

				// Propagate to Upper cell
				if (t4) {
					of1234[nPos.get_value(3)].set_value(3, f1234.get_value(3));
				}

				// Propagate to Lower-Right cell
				if (t1 && t2) {
					of5678[nPos.get_value(4)].set_value(0, f5678.get_value(0));
				}

				// Propogate to Lower-Left cell
				if (t2 && t3) {
					of5678[nPos.get_value(5)].set_value(1, f5678.get_value(1));
				}

				// Propagate to Upper-Left cell
				if (t3 && t4) {
					of5678[nPos.get_value(6)].set_value(2, f5678.get_value(2));
				}

				// Propagate to Upper-Right cell
				if (t4 && t1) {
					of5678[nPos.get_value(7)].set_value(3, f5678.get_value(3));
				}

				auto getColor = [](float2 inVelocity, bool isBoundary) {
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

						if (velocityMangitude < color1.get_value(3))
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

					return color * 255;
				};

				setPixelForNewLattice(getColor(u, type[pos]));

			});
		});
	}
	catch (cl::sycl::compile_program_error e)
	{
		qDebug() << e.what();
		std::exit(e.get_cl_code());
	}
	catch (cl::sycl::exception e)
	{
		qDebug() << e.what();
		std::exit(e.get_cl_code());
	}
	catch (std::exception e)
	{
		qDebug() << e.what();
		std::exit(EXIT_FAILURE);
	}

	CLcommandqueues().at(dev_id).enqueueReleaseGLObjects(&interop_resources, nullptr, &release);

	// Wait for all OpenCL commands to finish
	if (!cl_khr_gl_event_supported) cl::finish();
	else release.wait();


	// Swap front and back buffer handles
	std::swap(CL_latticeImages[Front], CL_latticeImages[Back]);
	std::swap(latticeImages[Front], latticeImages[Back]);
	std::swap(texs[Front], texs[Back]);
	// TEST swaps
	std::swap(f0_buffers[Buffer::Front], f0_buffers[Buffer::Back]);
	std::swap(f1234_buffers[Buffer::Front], f1234_buffers[Buffer::Back]);
	std::swap(f5678_buffers[Buffer::Front], f5678_buffers[Buffer::Back]);
	//testOutputs();

	imageDrawn = false;
}

void Raycaster::testOutputs() {
	auto f0 = f0_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read>();
	auto f1234 = f1234_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read>();
	auto f5678 = f5678_buffers[Buffer::Front]->get_access<cl::sycl::access::mode::read>();

	std::ofstream of0_file("of.txt");
	std::ofstream of1234_file("of1234.txt");
	std::ofstream of5678_file("of5678.txt");


	for (int i = 0; i < f0.get_count(); i++) {
		of0_file << (float)f0[i] << "\n";
	}

	for (int i = 0; i < f1234.get_count(); i++) {
		//qDebug() << f1234[i].get_value(0) << "\n";
		of1234_file << f1234[i].get_value(0) << "\n";
		of1234_file << f1234[i].get_value(1) << "\n";
		of1234_file << f1234[i].get_value(2) << "\n";
		of1234_file << f1234[i].get_value(3) << "\n";
	}

	for (int i = 0; i < f5678.get_count(); i++) {
		of5678_file << f5678[i].get_value(0) << "\n";
		of5678_file << f5678[i].get_value(1) << "\n";
		of5678_file << f5678[i].get_value(2) << "\n";
		of5678_file << f5678[i].get_value(3) << "\n";
	}

	of0_file.close();
	of1234_file.close();
	of5678_file.close();
}


// Raycaster
//void Raycaster::updateScene_3()
//{
//	// NOTE 1: When cl_khr_gl_event is NOT supported, then clFinish() is the only portable
//	//         sync method and hence that will be called.
//	//
//	// NOTE 2.1: When cl_khr_gl_event IS supported AND the possibly conflicting OpenGL
//	//           context is current to the thread, then it is sufficient to wait for events
//	//           of clEnqueueAcquireGLObjects, as the spec guarantees that all OpenGL
//	//           operations involving the acquired memory objects have finished. It also
//	//           guarantees that any OpenGL commands issued after clEnqueueReleaseGLObjects
//	//           will not execute until the release is complete.
//	//         
//	//           See: opencl-1.2-extensions.pdf (Rev. 15. Chapter 9.8.5)
//
//	cl::Event acquire, release;
//
//	CLcommandqueues().at(dev_id).enqueueAcquireGLObjects(&interop_resources, nullptr, &acquire);
//
//	try
//	{
//		// Start raymarch lambda
//		auto m_raymarch = [](const cl::sycl::float3& camPos, const cl::sycl::float3& rayDirection, const float startT, const float endT, const float deltaS)
//		{
//			int saturationThreshold = 0;
//			// example lambda functions that could be given by the user
//			// density function(spherical harminics) inside the extent
//			auto densityFunc = [=](const float& r, const float& theta, const float& /*phi*/)
//			{
//#ifdef __SYCL_DEVICE_ONLY__
//				float sqrt3fpi = cl::sycl::sqrt(3.0f / M_PI);
//				//float val = 1.0f / 2.0f * sqrt3fpi * cl::sycl::cos(theta + phiii); // Y(l = 1, m = 0)
//				float val = 1.0f / 2.0f * sqrt3fpi * cl::sycl::cos(theta); // Y(l = 1, m = 0)
//				float result = cl::sycl::fabs(2 * cl::sycl::fabs(val) - r);
//#else
//				float sqrt3fpi = 1.0f;
//				float val = 1.0f;
//				float result = 1.0f;
//
//				(void)sqrt3fpi;
//				(void)r;
//				(void)theta;
//#endif
//				if (result < 0.01f)	// thickness of shell 
//					return val < 0 ? -1 : 1;
//				else
//					return 0;
//			};
//
//			// color according to the incoming density
//			auto colorFunc = [](const int density)
//			{
//				if (density > 0)
//				{
//					return cl::sycl::float4(0, 0, 1, 0); // blue
//				}
//				else if (density < 0)
//				{
//					return cl::sycl::float4(1, 1, 0, 0); // yellow
//				}
//				else
//					return  cl::sycl::float4(0, 0, 0, 0); // black
//			};
//
//			cl::sycl::float4 finalColor(0.0f, 0.0f, 0.0f, 0.0f);
//			cl::sycl::float3 location(0.0f, 0.0f, 0.0f);
//
//			location = camPos + startT * rayDirection;
//
//			float current_t = startT;
//
//			while (current_t < endT)
//			{
//				location = location + deltaS * rayDirection;
//				current_t += deltaS;
//
//				// check if it is inside
//				//if (!IsOutside(location))
//				float x = location.x();
//				float y = location.y();
//				float z = location.z();
//				//if (x < extent.m_maxX)
//				//if ((x < extent.m_maxX) && y < (extent.m_maxY) && (z < extent.m_maxZ) &&
//				//	(x > extent.m_minX) && (y > extent.m_minY) && (z > extent.m_minZ))
//				//{
//					// Convert to spherical coordinated
//					//float r = sqrt(location.x*location.x + location.y*location.y + location.z*location.z);
//#ifdef __SYCL_DEVICE_ONLY__
//				float r = cl::sycl::length(location);
//				float theta = cl::sycl::acos(location.z() / r); // *180 / 3.1415926f; // convert to degrees?
//				float phi = cl::sycl::atan2(y, x); // *180 / 3.1415926f;
//#else
//				float r = 0.f;
//				float theta = 0.f;
//				float phi = 0.f;
//#endif
//
//				cl::sycl::float4 color = colorFunc(densityFunc(r, theta, phi));
//
//
//				finalColor += color;
//				//} // end if check isInside
//
//				// stop the ray, when color reaches the saturation.
//				if (finalColor.r() > saturationThreshold || finalColor.g() > saturationThreshold
//					|| finalColor.b() > saturationThreshold)
//					break;
//			}
//
//			// normalizer according to the highest rgb value
//			auto normalizer = std::max((float)1.0f, std::max(std::max(finalColor.r(), finalColor.g()), finalColor.b()));
//			finalColor /= normalizer;
//			finalColor *= 255;
//
//
//			return cl::sycl::float4(finalColor.r(), finalColor.g(), finalColor.b(), 255.f);
//		};
//		// END raymarch lambda
//
//
//		compute_queue.submit([&](cl::sycl::handler& cgh)
//		{
//			using namespace cl::sycl;
//
//			auto old_lattice = latticeImages[Buffer::Front]->get_access<float4, access::mode::read>(cgh);
//			auto new_lattice = latticeImages[Buffer::Back]->get_access<float4, access::mode::write>(cgh);
//
//			sampler periodic{ coordinate_normalization_mode::unnormalized,
//							  addressing_mode::none,
//							  filtering_mode::nearest };
//
//			auto aspectRatio = (float)old_lattice.get_range()[0] / old_lattice.get_range()[1];
//			//float scaleFOV = tan(120.f / 2 * M_PI / 180);
//			// scaleFOV?
//			cgh.parallel_for<kernels::RaycasterStep>(range<2>{ old_lattice.get_range() },
//				[=, ViewToWorldMtx = m_viewToWorldMtx, camPos = m_vecEye, sphereCenter = glm::vec3(0.f, 0.f, 0.f), sphereRadius2 = 1.96f, raymarch = m_raymarch, deltaS = 0.02f
//				](const item<2> i)
//			{
//				// Minden mehet a regivel, mert jelenleg nem kell az uv koordinate transzformalgatas
//				int2 pixelIndex = i.get_id();
//				auto getPixelFromOldLattice = [=](int2 in) { return old_lattice.read(in, periodic); };
//				auto setPixelForNewLattice = [=](float4 in) { new_lattice.write((int2)i.get_id(), in); };
//
//
//				glm::vec4 rayVec((2 * (i[0] + 0.5f) / (float)old_lattice.get_range()[0] - 1)* aspectRatio /* * scaleFOV */,
//					(1 - 2 * (i[1] + 0.5f) / (float)old_lattice.get_range()[1]) /* * scaleFOV*/,
//					-1.0f, 1.0f);
//
//				float t0 = -1E+36f;
//				float t1 = -1E+36f;
//
//				glm::vec3 transformedCamRayDir = glm::vec3(ViewToWorldMtx * rayVec) - camPos;
//#ifdef __SYCL_DEVICE_ONLY__
//				cl::sycl::float3 transformedCamRayDirFloat3 = cl::sycl::normalize(cl::sycl::float3{ transformedCamRayDir.x, transformedCamRayDir.y, transformedCamRayDir.z });
//#else
//				cl::sycl::float3 transformedCamRayDirFloat3;
//#endif
//
//				auto getIntersections_lambda = [&t0, &t1](const cl::sycl::float3 rayorig, const cl::sycl::float3 raydir, const cl::sycl::float3 sphereCenter,
//					const float sphereRadius2) {
//					cl::sycl::float3 l = sphereCenter - rayorig;
//					float tca = cl::sycl::dot(l, raydir);
//					float d2 = cl::sycl::dot(l, l) - tca * tca;
//
//					bool isIntersected = true;
//					if ((sphereRadius2 - d2) < 0.0001f) {
//						isIntersected = false;
//
//					}
//#ifdef __SYCL_DEVICE_ONLY__
//					float thc = cl::sycl::sqrt(sphereRadius2 - d2);
//#else
//					float thc = 0.f;
//#endif
//					t0 = tca - thc;
//					t1 = tca + thc;
//
//					return isIntersected;
//
//				};
//
//				auto camPosFloat3 = cl::sycl::float3(camPos.x, camPos.y, camPos.z);
//				auto bIntersected = getIntersections_lambda(camPosFloat3, transformedCamRayDirFloat3,
//					cl::sycl::float3(sphereCenter.x, sphereCenter.y, sphereCenter.z), sphereRadius2);
//
//				cl::sycl::float4 pixelColor;
//				if (bIntersected && t0 > 0.0 && t1 > 0.0)
//				{
//					//pixelColor = cl::sycl::float4(255, 0, 0, 255);
//					pixelColor = raymarch(camPosFloat3, transformedCamRayDirFloat3, t0, t1, deltaS);
//				}
//				// if we are inside the spehere, we trace from the the ray's original position
//				else if (bIntersected && t1 > 0.0)
//				{
//					//pixelColor = cl::sycl::float4(0, 255, 0, 255);
//					pixelColor = raymarch(camPosFloat3, transformedCamRayDirFloat3, 0.0, t1, deltaS);
//				}
//				else
//				{
//					pixelColor = cl::sycl::float4(0.f, 0.f, 0.f, 255.f);
//				}
//
//				// seting rgb value for every pixel
//				setPixelForNewLattice(pixelColor);
//			});
//		});
//	}
//	catch (cl::sycl::compile_program_error e)
//	{
//		qDebug() << e.what();
//		std::exit(e.get_cl_code());
//	}
//	catch (cl::sycl::exception e)
//	{
//		qDebug() << e.what();
//		std::exit(e.get_cl_code());
//	}
//	catch (std::exception e)
//	{
//		qDebug() << e.what();
//		std::exit(EXIT_FAILURE);
//	}
//
//	CLcommandqueues().at(dev_id).enqueueReleaseGLObjects(&interop_resources, nullptr, &release);
//
//	// Wait for all OpenCL commands to finish
//	if (!cl_khr_gl_event_supported) cl::finish();
//	else release.wait();
//
//	// Swap front and back buffer handles
//	std::swap(CL_latticeImages[Front], CL_latticeImages[Back]);
//	std::swap(latticeImages[Front], latticeImages[Back]);
//	std::swap(texs[Front], texs[Back]);
//
//	imageDrawn = false;
//}
//
//// Conway
//void Raycaster::updateScene_2()
//{
//	// NOTE 1: When cl_khr_gl_event is NOT supported, then clFinish() is the only portable
//	//         sync method and hence that will be called.
//	//
//	// NOTE 2.1: When cl_khr_gl_event IS supported AND the possibly conflicting OpenGL
//	//           context is current to the thread, then it is sufficient to wait for events
//	//           of clEnqueueAcquireGLObjects, as the spec guarantees that all OpenGL
//	//           operations involving the acquired memory objects have finished. It also
//	//           guarantees that any OpenGL commands issued after clEnqueueReleaseGLObjects
//	//           will not execute until the release is complete.
//	//         
//	//           See: opencl-1.2-extensions.pdf (Rev. 15. Chapter 9.8.5)
//
//	cl::Event acquire, release;
//
//	CLcommandqueues().at(dev_id).enqueueAcquireGLObjects(&interop_resources, nullptr, &acquire);
//
//	try
//	{
//		compute_queue.submit([&](cl::sycl::handler& cgh)
//		{
//			using namespace cl::sycl;
//
//			auto old_lattice = latticeImages[Buffer::Front]->get_access<float4, access::mode::read>(cgh);
//			auto new_lattice = latticeImages[Buffer::Back]->get_access<float4, access::mode::write>(cgh);
//
//			sampler periodic{ coordinate_normalization_mode::normalized,
//					addressing_mode::repeat,
//					filtering_mode::nearest };
//
//			float2 d = float2{ 1, 1 } / float2{ old_lattice.get_range()[0], old_lattice.get_range()[1] };
//
//
//			cgh.parallel_for<kernels::Test>(range<2>{ old_lattice.get_range() },
//				[=](const item<2> i)
//			{
//				// Convert unnormalized floating coords offsetted by self to normalized uv
//				auto uv = [=, s = float2{ i.get_id()[0], i.get_id()[1] }, d2 = d * 0.5f](float2 in) { return (s + in) * d + d2; };
//
//				auto old = [=](float2 in) { return old_lattice.read(uv(in), periodic).r() > 0.5f; };
//				auto next = [=](bool v) { new_lattice.write((int2)i.get_id(), float4{ v, v, v, 1.f }); };
//
//				std::array<bool, 8> neighbours = {
//					old(float2{ -1,+1 }), old(float2{ 0,+1 }), old(float2{ +1,+1 }),
//					old(float2{ -1,0 }),                     old(float2{ +1,0 }),
//					old(float2{ -1,-1 }), old(float2{ 0,-1 }), old(float2{ +1,-1 }) };
//
//				bool self = old(float2{ 0,0 });
//
//				auto count = std::count(neighbours.cbegin(), neighbours.cend(), true);
//
//				next(self ? (count < 2 || count > 3 ? 0.f : 1.f) : (count == 3 ? 1.f : 0.f));
//			});
//		});
//	}
//	catch (cl::sycl::compile_program_error e)
//	{
//		qDebug() << e.what();
//		std::exit(e.get_cl_code());
//	}
//	catch (cl::sycl::exception e)
//	{
//		qDebug() << e.what();
//		std::exit(e.get_cl_code());
//	}
//	catch (std::exception e)
//	{
//		qDebug() << e.what();
//		std::exit(EXIT_FAILURE);
//	}
//
//	CLcommandqueues().at(dev_id).enqueueReleaseGLObjects(&interop_resources, nullptr, &release);
//
//	// Wait for all OpenCL commands to finish
//	if (!cl_khr_gl_event_supported) cl::finish();
//	else release.wait();
//
//	// Swap front and back buffer handles
//	std::swap(CL_latticeImages[Front], CL_latticeImages[Back]);
//	std::swap(latticeImages[Front], latticeImages[Back]);
//	std::swap(texs[Front], texs[Back]);
//
//	imageDrawn = false;
//}