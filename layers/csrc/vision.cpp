// #include <torch/extension.h>
// #include <cuda_runtime_api.h>
#include <torch/types.h>
#include <torch/extension.h>

// #include "avgpool/aconv4.h"
#include "conv4/conv4.h"
#include "conv8/conv8.h"
#include "gconv4/gconv4.h"
#include "maxlen/maxlen.h"
namespace landmarkconv {


// #ifdef WITH_CUDA
// int get_cudart_version() {
//   return CUDART_VERSION;
// }
// #endif

// std::string get_cuda_version() {
// #ifdef WITH_CUDA
//   std::ostringstream oss;

//   // copied from
//   // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/CUDAHooks.cpp#L231
//   auto printCudaStyleVersion = [&](int v) {
//     oss << (v / 1000) << "." << (v / 10 % 100);
//     if (v % 10 != 0) {
//       oss << "." << (v % 10);
//     }
//   };
//   printCudaStyleVersion(get_cudart_version());
//   return oss.str();
// #else
//   return std::string("not available");
// #endif
// }

// // similar to
// // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Version.cpp
// std::string get_compiler_version() {
//   std::ostringstream ss;
// #if defined(__GNUC__)
// #ifndef __clang__
//   { ss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__; }
// #endif
// #endif

// #if defined(__clang_major__)
//   {
//     ss << "clang " << __clang_major__ << "." << __clang_minor__ << "."
//        << __clang_patchlevel__;
//   }
// #endif

// #if defined(_MSC_VER)
//   { ss << "MSVC " << _MSC_FULL_VER; }
// #endif
//   return ss.str();
// }




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    //guide conv
    m.def("G1_pool_forward", &G1_pool_forward, "G1 Top Left Pool forward");
    m.def("G1_pool_backward", &G1_pool_backward, "G1 Top Left Pool backward");
    m.def("G2_pool_forward", &G2_pool_forward, "G2 Top Left Pool forward");
    m.def("G2_pool_backward", &G2_pool_backward, "G2 Top Left Pool backward");
    m.def("G3_pool_forward", &G3_pool_forward, "G3 Top Left Pool forward");
    m.def("G3_pool_backward", &G3_pool_backward, "G3 Top Left Pool forward");
    m.def("G4_pool_forward", &G4_pool_forward, "G4 Top Left Pool forward");
    m.def("G4_pool_backward", &G4_pool_backward, "G4 Top Left Pool forward");

    // maxlen conv
    m.def("maxlen1_pool_forward", &maxlen1_pool_forward, "maxlen1 Top Left Pool forward");
    m.def("maxlen1_pool_backward", &maxlen1_pool_backward, "maxlen1 Top Left Pool backward");
    m.def("maxlen2_pool_forward", &maxlen2_pool_forward, "maxlen2 Top Right Pool forward");
    m.def("maxlen2_pool_backward", &maxlen2_pool_backward, "maxlen2 Top Right Pool backward"); 
    m.def("maxlen3_pool_forward", &maxlen3_pool_forward, "maxlen3 Bottom Left Pool forward");
    m.def("maxlen3_pool_backward", &maxlen3_pool_backward, "maxlen3 Bottom Left Pool backward"); 
    m.def("maxlen4_pool_forward", &maxlen4_pool_forward, "maxlen4 Bottom Right Pool forward");
    m.def("maxlen4_pool_backward", &maxlen4_pool_backward, "maxlen4 Bottom Right Pool backward"); 

    // // avgpool
    // m.def("R1_pool_forward", &R1_pool_forward, "R1 Top Left Pool forward");
    // m.def("R1_pool_backward", &R1_pool_backward, "R1 Top Left Pool backward");
    // m.def("R2_pool_forward", &R2_pool_forward, "R2 Top Left Pool forward");
    // m.def("R2_pool_backward", &R2_pool_backward, "R2 Top Left Pool backward");
    // m.def("R3_pool_forward", &R3_pool_forward, "R3 Top Left Pool forward");
    // m.def("R3_pool_backward", &R3_pool_backward, "R3 Top Left Pool forward");
    // m.def("R4_pool_forward", &R4_pool_forward, "R4 Top Left Pool forward");
    // m.def("R4_pool_backward", &R4_pool_backward, "R4 Top Left Pool forward");

    // conv4
    m.def("tl_pool_forward", &tl_pool_forward, "Top Left Pool fowardward");
    m.def("tl_pool_backward", &tl_pool_backward, "Top Left Pool backward");
    m.def("tr_pool_forward", &tr_pool_forward, "Top Right Pool fowardward");
    m.def("tr_pool_backward", &tr_pool_backward, "Top Right Pool backward");
    m.def("bl_pool_forward", &bl_pool_forward, "Bottom Left Pool fowardward");
    m.def("bl_pool_backward", &bl_pool_backward, "Bottom Left Pool backward");
    m.def("br_pool_forward", &br_pool_forward, "Bottom Right Left Pool fowardward");
    m.def("br_pool_backward", &br_pool_backward, "Bottom Right Pool backward");

    // conv8
    m.def("I1_pool_forward", &I1_pool_forward, "I1 Pool Forward");
    m.def("I1_pool_backward", &I1_pool_backward, "I1 Pool Backward"); 
    m.def("I2_pool_forward", &I2_pool_forward, "I2 Pool Forward");
    m.def("I2_pool_backward", &I2_pool_backward, "I2 Pool Backward"); 
    m.def("I3_pool_forward", &I3_pool_forward, "I3 Pool Forward");
    m.def("I3_pool_backward", &I3_pool_backward, "I3 Pool Backward"); 
    m.def("I4_pool_forward", &I4_pool_forward, "I4 Pool Forward");
    m.def("I4_pool_backward", &I4_pool_backward, "I4 Pool Backward"); 
    m.def("I5_pool_forward", &I5_pool_forward, "I5 Pool Forward");
    m.def("I5_pool_backward", &I5_pool_backward, "I5 Pool Backward"); 
    m.def("I6_pool_forward", &I6_pool_forward, "I6 Pool Forward");
    m.def("I6_pool_backward", &I6_pool_backward, "I6 Pool Backward"); 
    m.def("I7_pool_forward", &I7_pool_forward, "I7 Pool Forward");
    m.def("I7_pool_backward", &I7_pool_backward, "I7 Pool Backward"); 
    m.def("I8_pool_forward", &I8_pool_forward, "I8 Pool Forward");
    m.def("I8_pool_backward", &I8_pool_backward, "I8 Pool Backward"); 
}

} // namespace landmarkconv
