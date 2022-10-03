// #include <torch/torch.h>
// name should be different from .cpp file!!!
#include <torch/types.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>
#include <stdio.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
    i += blockDim.x * gridDim.x)

// #define THREADS_PER_BLOCK 1024
#define THREADS_PER_BLOCK 128

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65000;
  return min(optimal_block_num, max_block_num);
}

template <typename scalar_t>
__global__ void G1_forward_kernel(const int nthreads, //bs*ch
                                  const scalar_t *input_ptr, // value
                                  const scalar_t *guide_ptr, // query
                                  scalar_t *max_ptr, //guide.clone
                                  scalar_t *outptr,  //input.clone
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int dc = index % ch;
        int db = index / ch;
        for (int i = 0; i < sh; i++) {
            for (int j =  0; j < sw; j++) {
                auto x1 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto x2 = x1;
                auto x3 = x1;
                auto g1 = *(max_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto g2 = g1;
                auto g3 = g1;
                if (j > 0) g2 = *(max_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j-1);
                if (i > 0) g3 = *(max_ptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j);
                if (j > 0) x2 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + j-1);
                if (i > 0) x3 = *(outptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j);
                auto out = x1;
                auto guide = g1;
                if (guide < g2) {out = x2; guide = g2;}
                if (guide < g3) {out = x3; guide = g3;}
                outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = out;
                max_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = guide;
            }
        }
    }   
}

template <typename scalar_t>
__global__ void G1_backward_kernel(const int nthreads, 
                                  const scalar_t *input_ptr, 
                                  const scalar_t *guide_ptr,
                                  const scalar_t *output_ptr,
                                  const scalar_t *maxout_ptr,
                                  scalar_t *gradout_ptr,
                                  scalar_t *gradin_ptr,
                                  scalar_t *gradguide_ptr,
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int dc = index % ch;
        int db = index / ch;
        for (int i = sh-1; i >= 0; i--) {
            for (int j = sw-1; j >= 0; j--) {
                // auto x1 = *(input_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto gout = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto g1 = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto g2 = g1;
                auto g3 = g1;

                if (j > 0) g2 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j-1);
                if (i > 0) g3 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j);
                // printf("(%i, %i) g1 %f g2 %f g3 %f gout %f\n", i, j, g1, g2, g3, gout);
                g1 = scalar_t(g1 >= gout);
                g2 = (1-g1) * scalar_t(g2 >= gout);
                g3 = (1-g1) * (1-g2) * scalar_t(g3 >= gout);
                auto grad_x1 = g1;
                auto grad_x2 = g2;
                auto grad_x3 = g3;

                gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
                  grad_x1 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
                
                if (j > 0) gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j-1] += 
                  grad_x2 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
                if (i > 0) gradout_ptr[db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j] += 
                  grad_x3 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
            }
        }
    }
}


template <typename scalar_t>
__global__ void G2_forward_kernel(const int nthreads, 
                                  const scalar_t *input_ptr, 
                                  const scalar_t *guide_ptr,
                                  scalar_t *max_ptr,
                                  scalar_t *outptr,
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int dc = index % ch;
        int db = index / ch;
        for (int i = 0; i < sh; i++) {
            for (int j = sw-1; j >= 0; j--) {
                auto x1 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto x2 = x1;
                auto x3 = x1;
                auto g1 = *(max_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto g2 = g1;
                auto g3 = g1;
                if (j < sw-1) x2 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + j+1);
                if (i > 0) x3 = *(outptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j);
                if (j < sw-1) g2 = *(max_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j+1);
                if (i > 0) g3 = *(max_ptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j);
                auto out = x1;
                auto guide = g1;
                //softmax weight
                if (guide < g2) {out = x2; guide = g2;}
                if (guide < g3) {out = x3; guide = g3;}
                outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = out;
                max_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = guide;
            }
        }
    }   
}

template <typename scalar_t>
__global__ void G2_backward_kernel(const int nthreads, 
                                  const scalar_t *input_ptr, 
                                  const scalar_t *guide_ptr,
                                  const scalar_t *output_ptr,
                                  const scalar_t *maxout_ptr,
                                  scalar_t *gradout_ptr,
                                  scalar_t *gradin_ptr,
                                  scalar_t *gradguide_ptr,
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int dc = index % ch;
        int db = index / ch;
        for (int i = sh-1; i >= 0; i--) {
            for (int j = 0; j < sw; j++) {
                // auto x1 = *(input_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto gout = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto g1 = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto g2 = g1;
                auto g3 = g1;
                
                if (j < sw-1) g2 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j+1);
                if (i > 0) g3 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j);
                g1 = scalar_t(g1 >= gout);
                g2 = (1-g1) * scalar_t(g2 >= gout);
                g3 = (1-g1) * (1-g2) * scalar_t(g3 >= gout);
                auto grad_x1 = g1;
                auto grad_x2 = g2;
                auto grad_x3 = g3;

                gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
                  grad_x1 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
                
                if (j < sw-1) gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j+1] += 
                  grad_x2 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
                if (i > 0) gradout_ptr[db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j] += 
                  grad_x3 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
            }
        }
    }   
}

template <typename scalar_t>
__global__ void G3_forward_kernel(const int nthreads, 
                                  const scalar_t *input_ptr, 
                                  const scalar_t *guide_ptr,
                                  scalar_t *max_ptr,
                                  scalar_t *outptr,
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int dc = index % ch;
        int db = index / ch;
        for (int i = sh-1; i >= 0; i--) {
            for (int j = 0; j < sw; j++) {
                auto x1 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto x2 = x1;
                auto x3 = x1;
                auto g1 = *(max_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto g2 = g1;
                auto g3 = g1;
                if (j > 0) g2 = *(max_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j-1);
                if (i < sh-1) g3 = *(max_ptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j);
                if (j > 0) x2 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + j-1);
                if (i < sh-1) x3 = *(outptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j);
                auto out = x1;
                auto guide = g1;
                if (guide < g2) {out = x2; guide = g2;}
                if (guide < g3) {out = x3; guide = g3;}
                outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = out;
                max_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = guide;
            }
        }
    }   
}

template <typename scalar_t>
__global__ void G3_backward_kernel(const int nthreads, 
                                  const scalar_t *input_ptr, 
                                  const scalar_t *guide_ptr,
                                  const scalar_t *output_ptr,
                                  const scalar_t *maxout_ptr,
                                  scalar_t *gradout_ptr,
                                  scalar_t *gradin_ptr,
                                  scalar_t *gradguide_ptr,
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int dc = index % ch;
        int db = index / ch;
        for (int i = 0; i < sh; i++) {
            for (int j = sw-1; j >= 0; j--) {
                auto gout = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto g1 = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto g2 = g1;
                auto g3 = g1;
                
                if (j > 0) g2 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j-1);
                if (i < sh-1) g3 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j);
                
                g1 = scalar_t(g1 >= gout);
                g2 = (1-g1) * scalar_t(g2 >= gout);
                g3 = (1-g1) * (1-g2) * scalar_t(g3 >= gout);
                auto grad_x1 = g1;
                auto grad_x2 = g2;
                auto grad_x3 = g3;

                gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
                  grad_x1 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
                
                if (j > 0) gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j-1] += 
                  grad_x2 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
                if (i < sh-1) gradout_ptr[db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j] += 
                  grad_x3 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
            }
        }
    }   
}



template <typename scalar_t>
__global__ void G4_forward_kernel(const int nthreads, 
                                  const scalar_t *input_ptr, 
                                  const scalar_t *guide_ptr,
                                  scalar_t *max_ptr,
                                  scalar_t *outptr,
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int dc = index % ch;
        int db = index / ch;
        for (int i = sh-1; i >= 0; i--) {
            for (int j = sw-1; j >= 0; j--) {
                auto x1 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto x2 = x1;
                auto x3 = x1;
                auto g1 = *(max_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto g2 = g1;
                auto g3 = g1;

                if (j < sw-1) g2 = *(max_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j+1);
                if (i < sh-1) g3 = *(max_ptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j);
                if (j < sw-1) x2 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + j+1);
                if (i < sh-1) x3 = *(outptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j);

                auto out = x1;
                auto guide = g1;
                if (guide < g2) {out = x2; guide = g2;}
                if (guide < g3) {out = x3; guide = g3;}
                outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = out;
                max_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = guide;
            }
        }
    }   
}

template <typename scalar_t>
__global__ void G4_backward_kernel(const int nthreads, 
                                  const scalar_t *input_ptr, 
                                  const scalar_t *guide_ptr,
                                  const scalar_t *output_ptr,
                                  const scalar_t *maxout_ptr,
                                  scalar_t *gradout_ptr,
                                  scalar_t *gradin_ptr,
                                  scalar_t *gradguide_ptr,
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int dc = index % ch;
        int db = index / ch;
        for (int i = 0; i < sh; i++) {
            for (int j = 0; j < sw; j++) {
                auto gout = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto g1 = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto g2 = g1;
                auto g3 = g1;
                
                if (j < sw-1) g2 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j+1);
                if (i < sh-1) g3 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j);
                g1 = scalar_t(g1 >= gout);
                g2 = (1-g1) * scalar_t(g2 >= gout);
                g3 = (1-g1) * (1-g2) * scalar_t(g3 >= gout);
                auto grad_x1 = g1;
                auto grad_x2 = g2;
                auto grad_x3 = g3;

                gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
                  grad_x1 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
                
                if (j < sw-1) gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j+1] += 
                  grad_x2 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
                if (i < sh-1) gradout_ptr[db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j] += 
                  grad_x3 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
            }
        }
    }   
}


namespace landmarkconv {

std::vector<at::Tensor> G1_pool_forward_laucher(
    const at::Tensor &input, 
    const at::Tensor &guide) {
    // Ensure CUDA uses the input tensor device.
    at::DeviceGuard guard(input.device());
    AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");
    // printf("call by cuda...\n");

    cudaDeviceSynchronize(); // for print
    auto output = input.clone();
    auto maxout = guide.clone();
    int bs = input.size(0);
    int ch = input.size(1);
    int sh = input.size(2);
    int sw = input.size(3);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "G1_pool_forward_laucher", ([&] {
            const scalar_t *input_ptr = input.data_ptr<scalar_t>();
            const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
            scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
            scalar_t *output_ptr = output.data_ptr<scalar_t>();
            G1_forward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                          0, at::cuda::getCurrentCUDAStream()>>>(
                bs*ch,
                input_ptr,
                guide_ptr,
                max_ptr,
                output_ptr,
                bs, ch, sh, sw
            );
          }
        )
      );

    THCudaCheck(cudaGetLastError());
    return {
        output,
        maxout
    };
}


std::vector<at::Tensor> G1_pool_backward_laucher(
  const at::Tensor &input,
  const at::Tensor &guide,
  const at::Tensor &output,
  const at::Tensor &maxout,
  const at::Tensor &grad_output
) {
  // Ensure CUDA uses the input tensor device.
  at::DeviceGuard guard(input.device());
  AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");
  cudaDeviceSynchronize(); // for print
 
  auto grad_input = at::zeros_like(input);
  auto grad_guide = at::zeros_like(guide);
  auto gradout = grad_output.clone();

  int bs = input.size(0);
  int ch = input.size(1);
  int sh = input.size(2);
  int sw = input.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    input.scalar_type(), "G1_pool_backward_laucher", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
        const scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
        const scalar_t *output_ptr = output.data_ptr<scalar_t>();
        scalar_t *gradout_ptr = gradout.data_ptr<scalar_t>();
        scalar_t *gradin_ptr = grad_input.data_ptr<scalar_t>();
        scalar_t *gradguide_ptr = grad_guide.data_ptr<scalar_t>();


        G1_backward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                      0, at::cuda::getCurrentCUDAStream()>>>(
            bs*ch,
            input_ptr,
            guide_ptr,
            output_ptr,
            max_ptr,
            gradout_ptr, 
            gradin_ptr,
            gradguide_ptr,
            bs, ch, sh, sw
        );
      }
    )
  );

  THCudaCheck(cudaGetLastError());
  return {
    grad_input,
    grad_guide
  };
}


std::vector<at::Tensor> G2_pool_forward_laucher(
  const at::Tensor &input, 
  const at::Tensor &guide) {
  // Ensure CUDA uses the input tensor device.
  at::DeviceGuard guard(input.device());
  AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");
  // printf("call by cuda...\n");

  cudaDeviceSynchronize(); // for print
  auto output = input.clone();
  auto maxout = guide.clone();
  int bs = input.size(0);
  int ch = input.size(1);
  int sh = input.size(2);
  int sw = input.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "G2_pool_forward_laucher", ([&] {
          const scalar_t *input_ptr = input.data_ptr<scalar_t>();
          const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
          scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
          scalar_t *output_ptr = output.data_ptr<scalar_t>();
          G2_forward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                        0, at::cuda::getCurrentCUDAStream()>>>(
              bs*ch,
              input_ptr,
              guide_ptr,
              max_ptr,
              output_ptr,
              bs, ch, sh, sw
          );
        }
      )
    );

  THCudaCheck(cudaGetLastError());
  return {
      output,
      maxout
  };
}


std::vector<at::Tensor> G2_pool_backward_laucher(
  const at::Tensor &input,
  const at::Tensor &guide,
  const at::Tensor &output,
  const at::Tensor &maxout,
  const at::Tensor &grad_output
) {
  // Ensure CUDA uses the input tensor device.
  at::DeviceGuard guard(input.device());
  AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");
  cudaDeviceSynchronize(); // for print

  auto grad_input = at::zeros_like(input);
  auto grad_guide = at::zeros_like(guide);
  auto gradout = grad_output.clone();

  int bs = input.size(0);
  int ch = input.size(1);
  int sh = input.size(2);
  int sw = input.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    input.scalar_type(), "G2_pool_backward_laucher", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
        const scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
        const scalar_t *output_ptr = output.data_ptr<scalar_t>();
        scalar_t *gradout_ptr = gradout.data_ptr<scalar_t>();
        scalar_t *gradin_ptr = grad_input.data_ptr<scalar_t>();
        scalar_t *gradguide_ptr = grad_guide.data_ptr<scalar_t>();


        G2_backward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                      0, at::cuda::getCurrentCUDAStream()>>>(
            bs*ch,
            input_ptr,
            guide_ptr,
            output_ptr,
            max_ptr,
            gradout_ptr, 
            gradin_ptr,
            gradguide_ptr,
            bs, ch, sh, sw
          );
        }
      )
    );

  THCudaCheck(cudaGetLastError());
  return {
    grad_input, 
    grad_guide
  };
}


std::vector<at::Tensor> G3_pool_forward_laucher(
  const at::Tensor &input, 
  const at::Tensor &guide) {
  // Ensure CUDA uses the input tensor device.
  at::DeviceGuard guard(input.device());
  AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");
  // printf("call by cuda...\n");

  cudaDeviceSynchronize(); // for print
  auto output = input.clone();
  auto maxout = guide.clone();
  int bs = input.size(0);
  int ch = input.size(1);
  int sh = input.size(2);
  int sw = input.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "G3_pool_forward_laucher", ([&] {
          const scalar_t *input_ptr = input.data_ptr<scalar_t>();
          const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
          scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
          scalar_t *output_ptr = output.data_ptr<scalar_t>();
          G3_forward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                        0, at::cuda::getCurrentCUDAStream()>>>(
              bs*ch,
              input_ptr,
              guide_ptr,
              max_ptr,
              output_ptr,
              bs, ch, sh, sw
          );
        }
      )
    );

  THCudaCheck(cudaGetLastError());
  return {
      output,
      maxout
  };
}


std::vector<at::Tensor> G3_pool_backward_laucher(
  const at::Tensor &input,
  const at::Tensor &guide,
  const at::Tensor &output,
  const at::Tensor &maxout,
  const at::Tensor &grad_output
  ) {
  // Ensure CUDA uses the input tensor device.
  at::DeviceGuard guard(input.device());
  AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");
  cudaDeviceSynchronize(); // for print

  auto grad_input = at::zeros_like(input);
  auto grad_guide = at::zeros_like(guide);
  auto gradout = grad_output.clone();

  int bs = input.size(0);
  int ch = input.size(1);
  int sh = input.size(2);
  int sw = input.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    input.scalar_type(), "G3_pool_backward_laucher", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
        const scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
        const scalar_t *output_ptr = output.data_ptr<scalar_t>();
        scalar_t *gradout_ptr = gradout.data_ptr<scalar_t>();
        scalar_t *gradin_ptr = grad_input.data_ptr<scalar_t>();
        scalar_t *gradguide_ptr = grad_guide.data_ptr<scalar_t>();


        G3_backward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                      0, at::cuda::getCurrentCUDAStream()>>>(
            bs*ch,
            input_ptr,
            guide_ptr,
            output_ptr,
            max_ptr,
            gradout_ptr, 
            gradin_ptr,
            gradguide_ptr,
            bs, ch, sh, sw
        );
      }
    )
  );

  THCudaCheck(cudaGetLastError());
  return {
    grad_input, 
    grad_guide
  };
}



std::vector<at::Tensor> G4_pool_forward_laucher(
  const at::Tensor &input, 
  const at::Tensor &guide) {
  // Ensure CUDA uses the input tensor device.
  at::DeviceGuard guard(input.device());
  AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");
  // printf("call by cuda...\n");

  cudaDeviceSynchronize(); // for print
  auto output = input.clone();
  auto maxout = guide.clone();
  int bs = input.size(0);
  int ch = input.size(1);
  int sh = input.size(2);
  int sw = input.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "G4_pool_forward_laucher", ([&] {
          const scalar_t *input_ptr = input.data_ptr<scalar_t>();
          const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
          scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
          scalar_t *output_ptr = output.data_ptr<scalar_t>();
          G4_forward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                        0, at::cuda::getCurrentCUDAStream()>>>(
              bs*ch,
              input_ptr,
              guide_ptr,
              max_ptr,
              output_ptr,
              bs, ch, sh, sw
          );
        }
      )
    );
  
  THCudaCheck(cudaGetLastError());
  return {
      output,
      maxout
  };
}


std::vector<at::Tensor> G4_pool_backward_laucher(
  const at::Tensor &input,
  const at::Tensor &guide,
  const at::Tensor &output,
  const at::Tensor &maxout,
  const at::Tensor &grad_output
  ) {
  // Ensure CUDA uses the input tensor device.
  at::DeviceGuard guard(input.device());
  AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");
  cudaDeviceSynchronize(); // for print

  auto grad_input = at::zeros_like(input);
  auto grad_guide = at::zeros_like(guide);
  auto gradout = grad_output.clone();

  int bs = input.size(0);
  int ch = input.size(1);
  int sh = input.size(2);
  int sw = input.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    input.scalar_type(), "G4_pool_backward_laucher", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
        const scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
        const scalar_t *output_ptr = output.data_ptr<scalar_t>();
        scalar_t *gradout_ptr = gradout.data_ptr<scalar_t>();
        scalar_t *gradin_ptr = grad_input.data_ptr<scalar_t>();
        scalar_t *gradguide_ptr = grad_guide.data_ptr<scalar_t>();


        G4_backward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                      0, at::cuda::getCurrentCUDAStream()>>>(
            bs*ch,
            input_ptr,
            guide_ptr,
            output_ptr,
            max_ptr,
            gradout_ptr, 
            gradin_ptr,
            gradguide_ptr,
            bs, ch, sh, sw
        );
      }
    )
  );

  THCudaCheck(cudaGetLastError());
  return {
    grad_input, 
    grad_guide
  };
}

} // namespace landmarkconv2 