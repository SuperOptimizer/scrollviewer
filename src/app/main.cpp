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

  // Default to the local test volume; accept a path or URL argument, plus
  // an optional tifxyz segment directory (or URL prefix) as a second arg.
  // scrollviewer <volume> [tifxyz-dir] [surface-scale] [overlay-zarr]
  std::string source = "D:/scrolldata/PHerc0332-masked.zarr";
  std::string surface, overlay;
  double surfaceScale = 1.0;
  if (argc > 1) source = argv[1];
  if (argc > 2) surface = argv[2];
  if (argc > 3) surfaceScale = std::atof(argv[3]);
  if (argc > 4) overlay = argv[4];

  sv::app::ViewerWindow window(source, surface, surfaceScale, overlay);
  window.show();
  return app.exec();
}
