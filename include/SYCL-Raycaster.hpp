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

    // helper function
    virtual void resetScene() = 0;
    virtual void mouseDrag(QMouseEvent* event_in) = 0;  // Handle mouse dragging
    virtual void updateSceneImpl() = 0;  

    size_t getMeshSize();
    void swapBuffers();
    virtual void swapDataBuffers();
    virtual void writeOutputsToFile();

protected:
    enum Buffer
    {
        Front = 0,
        Back = 1
    };

    cl::sycl::queue compute_queue;          // CommandQueue


    QPoint mousePos;                        // Variables to enable dragging
    float dist, phi, theta;                 // Mouse polar coordinates
    std::array<std::unique_ptr<cl::sycl::image<2>>, 2> latticeImages;   // Simulation data images


private:

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

    bool imageDrawn;                        // Whether image has been drawn since last iteration
    bool needMatrixReset;                   // Whether matrices need to be reset in shaders

    void setMatrices();                     // Update shader matrices

	glm::mat4 m_viewToWorldMtx; // Identity
	glm::vec3 m_vecEye;
	glm::vec3 m_vecTarget;
	glm::vec3 m_vecUp;

	void setVRMatrices();

	bool rightMouseButtonPressed;           // Variables to enable dragging
};
