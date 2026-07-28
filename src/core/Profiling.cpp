#include "core/Profiling.h"

namespace sv {

ProfStages& profStages() {
  static ProfStages stages;
  return stages;
}

}  // namespace sv
