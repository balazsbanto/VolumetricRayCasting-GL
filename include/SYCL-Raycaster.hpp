#pragma once

// NBody configure
#include <SYCL-Raycaster-Config.hpp>

// C++ behavioral defines
#define _USE_MATH_DEFINES

// Base class include
#include <InteropWindow.hpp>

// SYCL include
#include <CL/sycl.hpp>

// Graphics utility includes
#include <QMatrix4x4>
#include <QVector>

#include <glm/ext.hpp>


// C++ includes
#include <array>        // std::array
#include <fstream>
#include <memory>
#include <future>
#include <random>
#include <memory>
#include <sstream>
#include <algorithm>
#include <memory>       // std::unique_ptr


namespace kernels { struct RaycasterStep; struct Test; struct Lbm; }


class Raycaster : public InteropWindow
{
    Q_OBJECT

public:

    explicit Raycaster(std::size_t plat,
                    std::size_t dev,
                    cl_bitfield type,
                    QWindow *parent = 0);
    ~Raycaster() = default;

    virtual void initializeGL() override;
    virtual void initializeCL() override;
    virtual void updateScene() override;
    //virtual void updateScene_2();
    //virtual void updateScene_3();
    virtual void render() override;
    virtual void render(QPainter* painter) override;
    virtual void resizeGL(QResizeEvent* event_in) override;
    virtual bool event(QEvent *event_in) override;

    // LBM D2Q9
    // (1/relaxation time) Related to viscosity 
    float omega = 1.2f;
    
    //  Distribution Buffers
    std::array < std::unique_ptr<cl::sycl::buffer<float, 1> >, 2 > f0_buffers;
    std::array < std::unique_ptr<cl::sycl::buffer<cl::sycl::float4, 1>>, 2 > f1234_buffers;
    std::array < std::unique_ptr<cl::sycl::buffer<cl::sycl::float4, 1>>, 2 > f5678_buffers;
    
    // Host vectors
    std::array < std::vector<float> , 2 > f0_host;
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

    // helper function
    void resetLBM();
    size_t getMeshSize();
    float computefEq(float weight, cl::sycl::float2 dir, float rho, cl::sycl::float2 velocity);
    void runOnCPU();
    void testOutputs();
    // END LBM

private:

    enum Buffer
    {
        Front = 0,
        Back = 1
    };

    std::size_t dev_id;

    // OpenGL related variables
    std::unique_ptr<QOpenGLShader> vs, fs;
    std::unique_ptr<QOpenGLShaderProgram> sp;
    std::unique_ptr<QOpenGLBuffer> vbo;
    std::unique_ptr<QOpenGLVertexArrayObject> vao;
    std::array<std::unique_ptr<QOpenGLTexture>, 2> texs;

    // OpenCL related variables
    std::array<cl::ImageGL, 2> CL_latticeImages;
    std::vector<cl::Memory> interop_resources;  // Bloat
    bool cl_khr_gl_event_supported;

    // SYCL related variables
    cl::sycl::context context;              // Context
    cl::sycl::device device;                // Device
    cl::sycl::queue compute_queue;          // CommandQueue

    std::array<std::unique_ptr<cl::sycl::image<2>>, 2> latticeImages;   // Simulation data images

    bool imageDrawn;                        // Whether image has been drawn since last iteration
    bool needMatrixReset;                   // Whether matrices need to be reset in shaders

    void setMatrices();                     // Update shader matrices

	glm::mat4 m_viewToWorldMtx; // Identity
	glm::vec3 m_vecEye;
	glm::vec3 m_vecTarget;
	glm::vec3 m_vecUp;

	void setVRMatrices();

	bool rightMouseButtonPressed;           // Variables to enable dragging
	QPoint mousePos;                        // Variables to enable dragging
	float dist, phi, theta;                 // Mouse polar coordinates
	void mouseDrag(QMouseEvent* event_in);  // Handle mouse dragging

};
