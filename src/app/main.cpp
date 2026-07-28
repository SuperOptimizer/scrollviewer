#include <QApplication>
#include <QSurfaceFormat>
#include <QVTKOpenGLNativeWidget.h>

#include "app/ViewerWindow.h"

// Force the discrete GPU on hybrid-graphics laptops.
extern "C" {
__declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

int main(int argc, char** argv) {
  QSurfaceFormat fmt = QVTKOpenGLNativeWidget::defaultFormat();
  fmt.setVersion(4, 5);
  fmt.setProfile(QSurfaceFormat::CoreProfile);
  QSurfaceFormat::setDefaultFormat(fmt);
  QApplication::setAttribute(Qt::AA_ShareOpenGLContexts);

  QApplication app(argc, argv);

  // Default to the local test volume; accept a path or URL argument.
  std::string source = "D:/scrolldata/PHerc0332-masked.zarr";
  if (argc > 1) source = argv[1];

  sv::app::ViewerWindow window(source);
  window.show();
  return app.exec();
}
