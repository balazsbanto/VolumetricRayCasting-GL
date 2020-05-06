// Behavioral defines
//
// GLM
#define GLM_ENABLE_EXPERIMENTAL

#include <SYCL-Raycaster.hpp>
#include <glm/gtx/transform.hpp>

#include <iterator>

//#include "glm/ext.hpp"
//#include "C:/Diplomamunka/vcpkg/installed/x64-windows/include/glm/ext.hpp"

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
	resetScene();

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
    if(!sp->bind()) qWarning("QRaycaster: Failed to bind shaderprogram");
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
    QString text("QRaycaster: ");
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
    mouseDragImpl(event_in);

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
        updateSceneImpl();
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
	swapBuffers();

    writeOutputsToFile();

	imageDrawn = false;
}

void Raycaster::swapBuffers() {
	std::swap(CL_latticeImages[Front], CL_latticeImages[Back]);
	std::swap(latticeImages[Front], latticeImages[Back]);
	std::swap(texs[Front], texs[Back]);

    swapDataBuffers();
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