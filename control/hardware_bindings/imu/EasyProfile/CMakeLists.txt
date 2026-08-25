# CMakeLists.txt for EasyProfile library

# Add a library called 'EasyProfile' with all its source files.
# 2026-08-22: STATIC (was SHARED) — linked into imu_nanobind so the module does
# not depend on a libEasyProfile.so that only ever existed in the build tree.
add_library(EasyProfile STATIC
  BasicTypes.h
  EasyObjectDictionary.cpp
  EasyObjectDictionary.h
  EasyProfile.cpp
  EasyProfile.h
  EasyProtocol.cpp
  EasyProtocol.h
  EasyQueue.h
)

# Set the include directory for EasyProfile
target_include_directories(EasyProfile PUBLIC .)
# PIC: the static archive is folded into a shared Python module.
set_target_properties(EasyProfile PROPERTIES POSITION_INDEPENDENT_CODE ON)
