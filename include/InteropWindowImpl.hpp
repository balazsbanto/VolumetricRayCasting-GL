#pragma once

// NBody configure
#include <InteropWindowImpl-Config.hpp>

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

struct ScreenSize {
    int width = 0;
    int height = 0;

    float aspectRatio() {
        return (float)width / height;
    };
};

class InteropWindowImpl : public InteropWindow
{
    Q_OBJECT

public:

    explicit InteropWindowImpl(std::size_t plat,
                    std::size_t dev,
                    cl_bitfield type,
                    QWindow *parent = 0);
    ~InteropWindowImpl() = default;

    virtual void initializeGL() override;
    virtual void initializeCL() override;
    virtual void updateScene() override;
    virtual void render() override;
    virtual void render(QPainter* painter) override;
    virtual void resizeGL(QResizeEvent* event_in) override;
    virtual bool event(QEvent *event_in) override;

    // helper function
    virtual void resetScene() {};
    virtual void mouseDragImpl(QMouseEvent* event_in) {};
    virtual void mouseWheelEventImpl(QWheelEvent* wheel_event) {};
    virtual void swapDataBuffers() {};
    virtual void writeOutputsToFile() {};
    virtual void updateSceneImpl() = 0;  

    size_t getNrOfPixels();
    void swapBuffers();
    void mouseDrag(QMouseEvent* event_in);  // Handle mouse dragging
    void mouseWheelEvent(QWheelEvent* wheel_event);  // Handle mouse dragging


protected:
    enum Buffer
    {
        Front = 0,
        Back = 1
    };

    cl::sycl::queue compute_queue;          // CommandQueue
    glm::mat4 m_viewToWorldMtx; // Identity
    glm::vec3 m_vecEye;


    QPoint mousePos;                        // Variables to enable dragging
    float dist, phi, theta;                 // Mouse polar coordinates
    std::array<std::unique_ptr<cl::sycl::image<2>>, 2> latticeImages;   // Simulation data images
    bool needMatrixReset;                   // Whether matrices need to be reset in shaders
    ScreenSize screenSize;

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

    void setMatrices();                     // Update shader matrices

	glm::vec3 m_vecTarget;
	glm::vec3 m_vecUp;

	void setVRMatrices();

	bool rightMouseButtonPressed;           // Variables to enable dragging
};
