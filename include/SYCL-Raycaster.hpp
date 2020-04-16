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
    virtual void updateScene_2();
    virtual void updateScene_3();
    virtual void render() override;
    virtual void render(QPainter* painter) override;
    virtual void resizeGL(QResizeEvent* event_in) override;
    virtual bool event(QEvent *event_in) override;

    // LBM

    void resetLBM();
    // Directions
    float e[9][2] = { {0,0}, {1,0}, {0,1}, {-1,0}, {0,-1}, {1,1}, {-1,1}, {-1,-1}, {1,-1} };

    // Weights
    std::vector<float> w { 4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0 };

    // Omega
    const cl_float omega = 1.2f;
    std::vector<float> rho;
    std::vector<cl_float2> u;
    std::vector<int> h_dirX { 0, 1, 0, -1,  0, 1, -1,  -1,  1};
    std::vector<int> h_dirY { 0, 0, 1,  0, -1, 1,  1,  -1, -1};
    std::vector<float> h_if0;
    std::vector<float> h_if1234;
    std::vector<float> h_if5678;

    // Device outputs
    std::vector<float> d_of0;
    std::vector<float> d_of1234;
    std::vector<float> d_of5678;
    std::vector<float> d_velocity;


    bool *h_type;

    // helper function
    size_t getMeshSize();
    size_t getNrOf_f();
    size_t getU_size();
    float computefEq(cl_float weight, float dir[2], float rho,
        cl_float2 velocity);
    size_t N = 4;
    size_t DIM = 2;

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
