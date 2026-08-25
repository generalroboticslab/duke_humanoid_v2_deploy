nanobind_add_module(
  motor_bindings
  NB_SHARED
  STABLE_ABI
  motor_bindings.cpp
  motor.hpp
  motor.cpp
  gl_motor.hpp
  gl_motor.cpp
)
set_target_properties(motor_bindings PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  # 2026-08-22: RUNPATH = $ORIGIN so the loader looks BESIDE the module for
  # libnanobind-abi3.so (the copy below) instead of in the build tree; the
  # module survives a deleted/moved control/build without LD_LIBRARY_PATH.
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
  BUILD_WITH_INSTALL_RPATH ON
)
# Copy libnanobind-abi3.so next to our .so so the wrapper can dlopen
# without setting LD_LIBRARY_PATH.
add_custom_command(TARGET motor_bindings POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:nanobind-abi3>
    ${CMAKE_CURRENT_SOURCE_DIR}/
)
target_link_libraries(motor_bindings PRIVATE Eigen3::Eigen)
# 2026-08-22: EasyProfile / CSerialPort links dropped — the motor sources include
# neither (CAN socket + fmt + Eigen only); they belong to the imu module.
target_link_libraries(motor_bindings PUBLIC fmt::fmt)
