// Qt5 includes
#include <QGuiApplication>
#include <QMessageLogger>
#include <QCommandLineParser>

// Custom made includes
#include <InteropWindowImpl.hpp>

// SYCL include
#ifdef _MSC_VER 
#pragma warning( push )
#pragma warning( disable : 4310 ) // Prevents warning about cast truncates constant value
#pragma warning( disable : 4100 ) // Prevents warning about unreferenced formal parameter
#endif
#include <CL/sycl.hpp>
#ifdef _MSC_VER 
#pragma warning( pop )
#endif

#include <LatticeBoltzmann2D.hpp>
#include <RaycasterLatticeBoltzmann2D.hpp>
#include <RaycasterLbm3D.hpp>
#include <SphericalHarmonicsRaycaster.hpp>
#include <CubeRaycaster.hpp>

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);
    QCoreApplication::setApplicationName("SYCL-GL Raycaster sample");
    QCoreApplication::setApplicationVersion("1.0");

    QCommandLineParser parser;
    parser.setApplicationDescription("Sample application demonstrating OpenCL-OpenGL interop");
    parser.addHelpOption();
    parser.addVersionOption();
    parser.addOptions({
        {{"p", "platform"}, "The index of the platform to use", "unsigned integral", "0"},
        {{"d", "device"}, "The index of the device to use", "unsigned integral", "0"},
        {{"t", "type"}, "Device type to use", "[cpu|gpu|acc]", "def"}
    });

    parser.process(app);

    cl_bitfield dev_type = CL_DEVICE_TYPE_DEFAULT;
    std::size_t plat_id = 0u, dev_id = 0u;

    if (!parser.value("platform").isEmpty()) plat_id = parser.value("platform").toULong();
    if (!parser.value("device").isEmpty()) dev_id = parser.value("device").toULong();
    //if(!parser.value("type").isEmpty())
    //{
    //    if(parser.value("type") == "cpu")
    //        dev_type = CL_DEVICE_TYPE_CPU;
    //    else if(parser.value("type") == "gpu")
    //        dev_type = CL_DEVICE_TYPE_GPU;
    //    else if(parser.value("type") == "acc")
    //        dev_type = CL_DEVICE_TYPE_ACCELERATOR;
    //    else
    //    {
    //        qFatal("SYCL-InteropWindowImpl: Invalid device type: valid values are [cpu|gpu|acc]. Using CL_DEVICE_TYPE_DEFAULT instead.");
    //    }
    //}

    dev_type = CL_DEVICE_TYPE_GPU;

    //RaycasterLbm3D raycaster(plat_id, dev_id, dev_type);
    //RaycasterLatticeBoltzmann2D raycaster(plat_id, dev_id, dev_type);
    LatticeBoltzmann2D raycaster(plat_id, dev_id, dev_type);
    //SphericalHarmonicsRaycaster raycaster(plat_id, dev_id, dev_type);
    //CubeRaycaster raycaster(plat_id, dev_id, dev_type);

    raycaster.setGeometry(QRect(0, 0, 256, 256));
    raycaster.setVisibility(QWindow::Windowed);
    //raycaster.setVisibility(QWindow::Maximized);

    raycaster.setAnimating(false);

    // Qt5 constructs
    QSurfaceFormat my_surfaceformat;
    
    // Setup desired format
    my_surfaceformat.setRenderableType(QSurfaceFormat::RenderableType::OpenGL);
    my_surfaceformat.setProfile(QSurfaceFormat::OpenGLContextProfile::CoreProfile);
    my_surfaceformat.setSwapBehavior(QSurfaceFormat::SwapBehavior::DoubleBuffer);
    my_surfaceformat.setOption(QSurfaceFormat::DebugContext);
    my_surfaceformat.setMajorVersion(3);
    my_surfaceformat.setMinorVersion(3);
    my_surfaceformat.setRedBufferSize(8);
    my_surfaceformat.setGreenBufferSize(8);
    my_surfaceformat.setBlueBufferSize(8);
    my_surfaceformat.setAlphaBufferSize(8);
    my_surfaceformat.setDepthBufferSize(24);
    my_surfaceformat.setStencilBufferSize(8);
    my_surfaceformat.setStereo(false);

    raycaster.setFormat(my_surfaceformat);

    return app.exec();
}
