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
__global__ void R1_forward_kernel(const int nthreads, 
                                  const scalar_t *input_ptr, 
                                  const scalar_t *guide_ptr,
                                  scalar_t *max_ptr,
                                  scalar_t *dominant_ptr,
                                  scalar_t *outptr,
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) { 
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int dc = index % ch;
    int db = index / ch;

    // sum forward
    for (int i = 0; i < sh; i++) {
      for (int j = 0; j < sw; j++) {
        auto x1 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
        scalar_t x2 = 0.0;
        scalar_t x3 = 0.0;
        scalar_t x4 = 0.0;
        if (j > 0) x2 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + j-1);
        if (i > 0) x3 = *(outptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j);
        if (i > 0 && j > 0) x4 = *(outptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j-1);

        auto sigmoid = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
        // calculate the mean values
        outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = x1 + sigmoid * (x2 + x3) - sigmoid * sigmoid * x4;
        max_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
      }
    }

    // average forward
    for (int i = 0; i < sh; i++) {
      for (int j = 0; j < sw; j++) {
        auto sigmoid = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
        // dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 1;
        scalar_t x2 = 0.0;
        scalar_t x3 = 0.0;
        scalar_t x4 = 0.0;
        if (j > 0) x2 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j-1];
        if (i > 0) x3 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j];
        if (i > 0 && j > 0) x4 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j-1];
        dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 1 + sigmoid * x2 + sigmoid * x3 - sigmoid * sigmoid * x4;
        outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] /= dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        // printf("(%i, %i) d %f \n", i, j, dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j]);
      }
    }
  }
}


template <typename scalar_t>
__global__ void R1_backward_kernel(const int nthreads, 
                                  const scalar_t *input_ptr, 
                                  const scalar_t *guide_ptr,
                                  const scalar_t *output_ptr,
                                  const scalar_t *maxout_ptr,
                                  const scalar_t *dominant_ptr,
                                  scalar_t *gradout_ptr,
                                  scalar_t *gradin_ptr,
                                  scalar_t *gradguide_ptr,
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) {

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int dc = index % ch;
    int db = index / ch;

    // average backward
    for (int i = sh-1; i >= 0; i--) {
      for (int j = sw-1; j >= 0; j--) {
        auto sumout = maxout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        auto norm1 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        auto sigmoid = guide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        scalar_t norm2 = 0.0;
        scalar_t norm3 = 0.0;
        scalar_t norm4 = 0.0;
        if (j > 0) norm2 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j-1];
        if (i > 0) norm3 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j];
        if (i > 0 && j > 0) norm4 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j-1];
        // gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] /= dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] =  gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] / norm1;
        gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] += - sumout / ( norm1 * norm1) * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        
        if (j > 0) gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j-1] += sigmoid * gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        if (i > 0) gradguide_ptr[db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j] += sigmoid * gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        if (i > 0 && j > 0) gradguide_ptr[db * ch * sh * sw + dc * sh * sw + (i-1) * sw + (j-1)] += -sigmoid * sigmoid * gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] *= (norm2 + norm3) - 2 * sigmoid * norm4; // 
        // printf("(%i, %i) d %f gradguide %f \n", i, j, norm1, gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j]);
      }
    }

    // sum backward
    for (int i = sh-1; i >= 0; i--) {
      for (int j = sw-1; j >= 0; j--) {
        auto x1 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
        auto sigmoid = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
        scalar_t x2 = 0.0;
        scalar_t x3 = 0.0;
        scalar_t x4 = 0.0;
        if (j > 0) x2 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j-1);
        if (i > 0) x3 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j);
        if (i > 0 && j > 0) x4 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j-1);

        auto grad_x1 = 1;
        auto grad_x2 = sigmoid;
        auto grad_x3 = sigmoid;
        auto grad_x4 = -sigmoid*sigmoid;
        auto grad_sigmoid = x2 + x3 - 2 * sigmoid * x4;
        
        gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] += 
          grad_sigmoid * gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        
        if (j > 0) gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j-1] += 
          grad_x2 * gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        if (i > 0) gradin_ptr[db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j] += 
          grad_x3 * gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        if (i > 0 && j > 0) gradin_ptr[db * ch * sh * sw + dc * sh * sw + (i-1) * sw + (j-1)] += 
          grad_x4 * gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        
        gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
          grad_x1 * gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
      }
    }
  }
}


template <typename scalar_t>
__global__ void R2_forward_kernel(const int nthreads, 
                                  const scalar_t *input_ptr, 
                                  const scalar_t *guide_ptr,
                                  scalar_t *max_ptr,
                                  scalar_t *dominant_ptr,
                                  scalar_t *outptr,
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) { 
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int dc = index % ch;
    int db = index / ch;

    // sum forward
    for (int i = 0; i < sh; i++) {
      for (int j = sw-1; j >= 0; j--) {
        auto x1 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
        scalar_t x2 = 0.0;
        scalar_t x3 = 0.0;
        scalar_t x4 = 0.0;
        if (j < sw-1) x2 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + j+1);
        if (i > 0) x3 = *(outptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j);
        if (i > 0 && j < sw-1) x4 = *(outptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j+1);

        auto sigmoid = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
        // calculate the mean values
        outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = x1 + sigmoid * (x2 + x3) - sigmoid * sigmoid * x4;
        max_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
      }
    }

    // average forward
    for (int i = 0; i < sh; i++) {
      for (int j = sw-1; j >= 0; j--) {
        auto sigmoid = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
        // dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 1;
        scalar_t x2 = 0.0;
        scalar_t x3 = 0.0;
        scalar_t x4 = 0.0;
        if (j < sw-1) x2 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j+1];
        if (i > 0) x3 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j];
        if (i > 0 && j < sw-1) x4 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j+1];
        dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 1 + sigmoid * x2 + sigmoid * x3 - sigmoid * sigmoid * x4;
        outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] /= dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        // printf("(%i, %i) d %f \n", i, j, dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j]);
      }
    }
  }
}


template <typename scalar_t>
__global__ void R2_backward_kernel(const int nthreads, 
                                  const scalar_t *input_ptr, 
                                  const scalar_t *guide_ptr,
                                  const scalar_t *output_ptr,
                                  const scalar_t *maxout_ptr,
                                  const scalar_t *dominant_ptr,
                                  scalar_t *gradout_ptr,
                                  scalar_t *gradin_ptr,
                                  scalar_t *gradguide_ptr,
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) {

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int dc = index % ch;
    int db = index / ch;

    // average backward
    for (int i = sh-1; i >= 0; i--) {
      for (int j = 0; j < sw; j++) {
        auto sumout = maxout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        auto norm1 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        auto sigmoid = guide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        scalar_t norm2 = 0.0;
        scalar_t norm3 = 0.0;
        scalar_t norm4 = 0.0;
        if (j < sw-1) norm2 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j+1];
        if (i > 0) norm3 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j];
        if (i > 0 && j < sw-1) norm4 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j+1];
        // gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] /= dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] =  gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] / norm1;
        gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] += - sumout / ( norm1 * norm1) * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        
        if (j < sw-1) gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j+1] += sigmoid * gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        if (i > 0) gradguide_ptr[db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j] += sigmoid * gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        if (i > 0 && j < sw-1) gradguide_ptr[db * ch * sh * sw + dc * sh * sw + (i-1) * sw + (j+1)] += -sigmoid * sigmoid * gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] *= (norm2 + norm3) - 2 * sigmoid * norm4; // 
        // printf("(%i, %i) d %f gradguide %f \n", i, j, norm1, gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j]);
      }
    }

    // sum backward
    for (int i = sh-1; i >= 0; i--) {
      for (int j = 0; j < sw; j++) {
        auto x1 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
        auto sigmoid = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
        scalar_t x2 = 0.0;
        scalar_t x3 = 0.0;
        scalar_t x4 = 0.0;
        if (j < sw-1) x2 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j+1);
        if (i > 0) x3 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j);
        if (i > 0 && j < sw-1) x4 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j+1);

        auto grad_x1 = 1;
        auto grad_x2 = sigmoid;
        auto grad_x3 = sigmoid;
        auto grad_x4 = -sigmoid*sigmoid;
        auto grad_sigmoid = x2 + x3 - 2 * sigmoid * x4;
        
        gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] += 
          grad_sigmoid * gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        
        if (j < sw-1) gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j+1] += 
          grad_x2 * gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        if (i > 0) gradin_ptr[db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j] += 
          grad_x3 * gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        if (i > 0 && j < sw-1) gradin_ptr[db * ch * sh * sw + dc * sh * sw + (i-1) * sw + (j+1)] += 
          grad_x4 * gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        
        gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
          grad_x1 * gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
      }
    }
  }
}


template <typename scalar_t>
__global__ void R3_forward_kernel(const int nthreads, 
                                  const scalar_t *input_ptr, 
                                  const scalar_t *guide_ptr,
                                  scalar_t *max_ptr,
                                  scalar_t *dominant_ptr,
                                  scalar_t *outptr,
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) { 
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int dc = index % ch;
    int db = index / ch;

    // sum forward
    for (int i = sh-1; i >= 0; i--) {
      for (int j = 0; j < sw; j++) {
        auto x1 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
        scalar_t x2 = 0.0;
        scalar_t x3 = 0.0;
        scalar_t x4 = 0.0;
        if (j > 0) x2 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw+ j-1);
        if (i < sh-1) x3 = *(outptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j);
        if (i < sh-1 && j > 0) x4 = *(outptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j-1);

        auto sigmoid = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
        // calculate the mean values
        outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = x1 + sigmoid * (x2 + x3) - sigmoid * sigmoid * x4;
        max_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
      }
    }

    // average forward
    for (int i = sh-1; i >= 0; i--) {
      for (int j = 0; j < sw; j++) {
        auto sigmoid = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
        // dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 1;
        scalar_t x2 = 0.0;
        scalar_t x3 = 0.0;
        scalar_t x4 = 0.0;
        if (j > 0) x2 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j-1];
        if (i < sh-1) x3 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j];
        if (i < sh-1 && j > 0) x4 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j-1];
        dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 1 + sigmoid * x2 + sigmoid * x3 - sigmoid * sigmoid * x4;
        outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] /= dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        // printf("(%i, %i) d %f \n", i, j, dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j]);
      }
    }
  }
}


template <typename scalar_t>
__global__ void R3_backward_kernel(const int nthreads, 
                                  const scalar_t *input_ptr, 
                                  const scalar_t *guide_ptr,
                                  const scalar_t *output_ptr,
                                  const scalar_t *maxout_ptr,
                                  const scalar_t *dominant_ptr,
                                  scalar_t *gradout_ptr,
                                  scalar_t *gradin_ptr,
                                  scalar_t *gradguide_ptr,
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) {

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int dc = index % ch;
    int db = index / ch;
    // average backward
    for (int i = 0; i < sh; i++) {
      for (int j = sw-1; j >= 0; j--) {
        auto sumout = maxout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        auto norm1 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        auto sigmoid = guide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        scalar_t norm2 = 0.0;
        scalar_t norm3 = 0.0;
        scalar_t norm4 = 0.0;
        if (j > 0) norm2 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j-1];
        if (i < sh-1) norm3 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j];
        if (i < sh-1 && j > 0) norm4 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j-1];
        // gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] /= dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] =  gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] / norm1;
        gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] += - sumout / ( norm1 * norm1) * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        
        if (j > 0) gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j-1] += sigmoid * gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        if (i < sh-1) gradguide_ptr[db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j] += sigmoid * gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        if (i < sh-1 && j > 0) gradguide_ptr[db * ch * sh * sw + dc * sh * sw + (i+1) * sw + (j-1)] += -sigmoid * sigmoid * gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] *= (norm2 + norm3) - 2 * sigmoid * norm4;
      }
    }

    // sum backward
    for (int i = 0; i < sh; i++) {
      for (int j = sw-1; j >= 0; j--) {
        auto x1 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
        auto sigmoid = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
        scalar_t x2 = 0.0;
        scalar_t x3 = 0.0;
        scalar_t x4 = 0.0;
        if (j > 0) x2 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j-1);
        if (i < sh-1) x3 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j);
        if (i < sh-1 && j > 0) x4 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j-1);

        auto grad_x1 = 1;
        auto grad_x2 = sigmoid;
        auto grad_x3 = sigmoid;
        auto grad_x4 = -sigmoid*sigmoid;
        auto grad_sigmoid = x2 + x3 - 2 * sigmoid * x4;
        
        gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] += 
          grad_sigmoid * gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        
        if (j > 0) gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j-1] += 
          grad_x2 * gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        if (i < sh-1) gradin_ptr[db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j] += 
          grad_x3 * gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        if (i < sh-1 && j > 0) gradin_ptr[db * ch * sh * sw + dc * sh * sw + (i+1) * sw + (j-1)] += 
          grad_x4 * gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        
        gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
          grad_x1 * gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
      }
    }
  }
}


template <typename scalar_t>
__global__ void R4_forward_kernel(const int nthreads, 
                                  const scalar_t *input_ptr, 
                                  const scalar_t *guide_ptr,
                                  scalar_t *max_ptr,
                                  scalar_t *dominant_ptr,
                                  scalar_t *outptr,
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) { 
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int dc = index % ch;
    int db = index / ch;
    // sum forward
    for (int i = sh-1; i >= 0; i--) {
      for (int j = sw-1; j >= 0; j--) {
        auto x1 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
        scalar_t x2 = 0.0;
        scalar_t x3 = 0.0;
        scalar_t x4 = 0.0;
        if (j < sw-1) x2 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw+ j+1);
        if (i < sh-1) x3 = *(outptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j);
        if (i < sh-1 && j < sw-1) x4 = *(outptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j+1);

        auto sigmoid = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
        // calculate the mean values
        outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = x1 + sigmoid * (x2 + x3) - sigmoid * sigmoid * x4;
        max_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
      }
    }

    // average forward
    for (int i = sh-1; i >= 0; i--) {
      for (int j = sw-1; j >= 0; j--) {
        auto sigmoid = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
        // dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 1;
        scalar_t x2 = 0.0;
        scalar_t x3 = 0.0;
        scalar_t x4 = 0.0;
        if (j < sw-1) x2 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j+1];
        if (i < sh-1) x3 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j];
        if (i < sh-1 && j < sw-1) x4 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j+1];
        dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 1 + sigmoid * x2 + sigmoid * x3 - sigmoid * sigmoid * x4;
        outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] /= dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        // printf("(%i, %i) d %f \n", i, j, dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j]);
      }
    }
  }
}


template <typename scalar_t>
__global__ void R4_backward_kernel(const int nthreads, 
                                  const scalar_t *input_ptr, 
                                  const scalar_t *guide_ptr,
                                  const scalar_t *output_ptr,
                                  const scalar_t *maxout_ptr,
                                  const scalar_t *dominant_ptr,
                                  scalar_t *gradout_ptr,
                                  scalar_t *gradin_ptr,
                                  scalar_t *gradguide_ptr,
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) {

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int dc = index % ch;
    int db = index / ch;
    // average backward
    for (int i = 0; i < sh; i++) {
      for (int j = 0; j < sw; j++) {
        auto sumout = maxout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        auto norm1 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        auto sigmoid = guide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        scalar_t norm2 = 0.0;
        scalar_t norm3 = 0.0;
        scalar_t norm4 = 0.0;
        if (j < sw-1) norm2 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j+1];
        if (i < sh-1) norm3 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j];
        if (i < sh-1 && j < sw-1) norm4 = dominant_ptr[db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j+1];
        // gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] /= dominant_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] =  gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] / norm1;
        gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] += - sumout / ( norm1 * norm1) * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        
        if (j < sw-1) gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j+1] += sigmoid * gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        if (i < sh-1) gradguide_ptr[db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j] += sigmoid * gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        if (i < sh-1 && j < sw-1) gradguide_ptr[db * ch * sh * sw + dc * sh * sw + (i+1) * sw + (j+1)] += -sigmoid * sigmoid * gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] *= (norm2 + norm3) - 2 * sigmoid * norm4;
      }
    }

    // sum backward
    for (int i = 0; i < sh; i++) {
      for (int j = 0; j < sw; j++) {
        auto x1 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
        auto sigmoid = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
        scalar_t x2 = 0.0;
        scalar_t x3 = 0.0;
        scalar_t x4 = 0.0;
        if (j < sw-1) x2 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j+1);
        if (i < sh-1) x3 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j);
        if (i < sh-1 && j < sw-1) x4 = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j+1);

        auto grad_x1 = 1;
        auto grad_x2 = sigmoid;
        auto grad_x3 = sigmoid;
        auto grad_x4 = -sigmoid*sigmoid;
        auto grad_sigmoid = x2 + x3 - 2 * sigmoid * x4;
        
        gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] += 
          grad_sigmoid * gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        
        if (j < sw-1) gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j+1] += 
          grad_x2 * gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        if (i < sh-1) gradin_ptr[db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j] += 
          grad_x3 * gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        if (i < sh-1 && j < sh-1) gradin_ptr[db * ch * sh * sw + dc * sh * sw + (i+1) * sw + (j+1)] += 
          grad_x4 * gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
        
        gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
          grad_x1 * gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
      }
    }
  }
}





namespace landmarkconv {

std::vector<at::Tensor> R1_pool_forward_laucher(
    const at::Tensor &input, 
    const at::Tensor &guide) {
    // Ensure CUDA uses the input tensor device.
    at::DeviceGuard guard(input.device());
    AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");
  
    cudaDeviceSynchronize(); // for print
    auto output = input.clone();
    auto maxout = input.clone();
    auto dominant = at::zeros_like(input);
    int bs = input.size(0);
    int ch = input.size(1);
    int sh = input.size(2);
    int sw = input.size(3);
  
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "R1_pool_forward_laucher", ([&] {
            const scalar_t *input_ptr = input.data_ptr<scalar_t>();
            const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
            scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
            scalar_t *output_ptr = output.data_ptr<scalar_t>();
            scalar_t *dominant_ptr = dominant.data_ptr<scalar_t>();
            R1_forward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                          0, at::cuda::getCurrentCUDAStream()>>>(
                bs*ch,
                input_ptr,
                guide_ptr,
                max_ptr,
                dominant_ptr,
                output_ptr,
                bs, ch, sh, sw
            );
          }
        )
      );
  
    THCudaCheck(cudaGetLastError());
    return {
        output,
        maxout,
        dominant
    };
  }
  
  
std::vector<at::Tensor> R1_pool_backward_laucher(
  const at::Tensor &input,
  const at::Tensor &guide,
  const at::Tensor &output,
  const at::Tensor &maxout,
  const at::Tensor &dominant,
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
    input.scalar_type(), "R1_pool_backward_laucher", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
        const scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
        const scalar_t *output_ptr = output.data_ptr<scalar_t>();
        const scalar_t *dominant_ptr = dominant.data_ptr<scalar_t>();
        scalar_t *gradout_ptr = gradout.data_ptr<scalar_t>();
        scalar_t *gradin_ptr = grad_input.data_ptr<scalar_t>();
        scalar_t *gradguide_ptr = grad_guide.data_ptr<scalar_t>();


        R1_backward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                      0, at::cuda::getCurrentCUDAStream()>>>(
            bs*ch,
            input_ptr,
            guide_ptr,
            output_ptr,
            max_ptr,
            dominant_ptr,
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

std::vector<at::Tensor> R2_pool_forward_laucher(
  const at::Tensor &input, 
  const at::Tensor &guide) {
  // Ensure CUDA uses the input tensor device.
  at::DeviceGuard guard(input.device());
  AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");

  cudaDeviceSynchronize(); // for print
  auto output = input.clone();
  auto maxout = input.clone();
  auto dominant = at::zeros_like(input);
  int bs = input.size(0);
  int ch = input.size(1);
  int sh = input.size(2);
  int sw = input.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "R1_pool_forward_laucher", ([&] {
          const scalar_t *input_ptr = input.data_ptr<scalar_t>();
          const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
          scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
          scalar_t *output_ptr = output.data_ptr<scalar_t>();
          scalar_t *dominant_ptr = dominant.data_ptr<scalar_t>();
          R2_forward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                        0, at::cuda::getCurrentCUDAStream()>>>(
              bs*ch,
              input_ptr,
              guide_ptr,
              max_ptr,
              dominant_ptr,
              output_ptr,
              bs, ch, sh, sw
          );
        }
      )
    );

  THCudaCheck(cudaGetLastError());
  return {
      output,
      maxout,
      dominant
  };
}


std::vector<at::Tensor> R2_pool_backward_laucher(
  const at::Tensor &input,
  const at::Tensor &guide,
  const at::Tensor &output,
  const at::Tensor &maxout,
  const at::Tensor &dominant,
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
  input.scalar_type(), "R1_pool_backward_laucher", ([&] {
      const scalar_t *input_ptr = input.data_ptr<scalar_t>();
      const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
      const scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
      const scalar_t *output_ptr = output.data_ptr<scalar_t>();
      const scalar_t *dominant_ptr = dominant.data_ptr<scalar_t>();
      scalar_t *gradout_ptr = gradout.data_ptr<scalar_t>();
      scalar_t *gradin_ptr = grad_input.data_ptr<scalar_t>();
      scalar_t *gradguide_ptr = grad_guide.data_ptr<scalar_t>();


      R2_backward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                    0, at::cuda::getCurrentCUDAStream()>>>(
          bs*ch,
          input_ptr,
          guide_ptr,
          output_ptr,
          max_ptr,
          dominant_ptr,
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

std::vector<at::Tensor> R3_pool_forward_laucher(
  const at::Tensor &input, 
  const at::Tensor &guide) {
  // Ensure CUDA uses the input tensor device.
  at::DeviceGuard guard(input.device());
  AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");

  cudaDeviceSynchronize(); // for print
  auto output = input.clone();
  auto maxout = input.clone();
  auto dominant = at::zeros_like(input);
  int bs = input.size(0);
  int ch = input.size(1);
  int sh = input.size(2);
  int sw = input.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "R3_pool_forward_laucher", ([&] {
          const scalar_t *input_ptr = input.data_ptr<scalar_t>();
          const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
          scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
          scalar_t *output_ptr = output.data_ptr<scalar_t>();
          scalar_t *dominant_ptr = dominant.data_ptr<scalar_t>();
          R3_forward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                        0, at::cuda::getCurrentCUDAStream()>>>(
              bs*ch,
              input_ptr,
              guide_ptr,
              max_ptr,
              dominant_ptr,
              output_ptr,
              bs, ch, sh, sw
          );
        }
      )
    );

  THCudaCheck(cudaGetLastError());
  return {
      output,
      maxout,
      dominant
  };
}


std::vector<at::Tensor> R3_pool_backward_laucher(
  const at::Tensor &input,
  const at::Tensor &guide,
  const at::Tensor &output,
  const at::Tensor &maxout,
  const at::Tensor &dominant,
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
    input.scalar_type(), "R3_pool_backward_laucher", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
        const scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
        const scalar_t *output_ptr = output.data_ptr<scalar_t>();
        const scalar_t *dominant_ptr = dominant.data_ptr<scalar_t>();
        scalar_t *gradout_ptr = gradout.data_ptr<scalar_t>();
        scalar_t *gradin_ptr = grad_input.data_ptr<scalar_t>();
        scalar_t *gradguide_ptr = grad_guide.data_ptr<scalar_t>();


        R3_backward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                      0, at::cuda::getCurrentCUDAStream()>>>(
            bs*ch,
            input_ptr,
            guide_ptr,
            output_ptr,
            max_ptr,
            dominant_ptr,
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

std::vector<at::Tensor> R4_pool_forward_laucher(
  const at::Tensor &input, 
  const at::Tensor &guide) {
  // Ensure CUDA uses the input tensor device.
  at::DeviceGuard guard(input.device());
  AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");

  cudaDeviceSynchronize(); // for print
  auto output = input.clone();
  auto maxout = input.clone();
  auto dominant = at::zeros_like(input);
  int bs = input.size(0);
  int ch = input.size(1);
  int sh = input.size(2);
  int sw = input.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "R1_pool_forward_laucher", ([&] {
          const scalar_t *input_ptr = input.data_ptr<scalar_t>();
          const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
          scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
          scalar_t *output_ptr = output.data_ptr<scalar_t>();
          scalar_t *dominant_ptr = dominant.data_ptr<scalar_t>();
          R4_forward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                        0, at::cuda::getCurrentCUDAStream()>>>(
              bs*ch,
              input_ptr,
              guide_ptr,
              max_ptr,
              dominant_ptr,
              output_ptr,
              bs, ch, sh, sw
          );
        }
      )
    );

  THCudaCheck(cudaGetLastError());
  return {
      output,
      maxout,
      dominant
  };
}


std::vector<at::Tensor> R4_pool_backward_laucher(
  const at::Tensor &input,
  const at::Tensor &guide,
  const at::Tensor &output,
  const at::Tensor &maxout,
  const at::Tensor &dominant,
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
    input.scalar_type(), "R1_pool_backward_laucher", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
        const scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
        const scalar_t *output_ptr = output.data_ptr<scalar_t>();
        const scalar_t *dominant_ptr = dominant.data_ptr<scalar_t>();
        scalar_t *gradout_ptr = gradout.data_ptr<scalar_t>();
        scalar_t *gradin_ptr = grad_input.data_ptr<scalar_t>();
        scalar_t *gradguide_ptr = grad_guide.data_ptr<scalar_t>();


        R4_backward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                      0, at::cuda::getCurrentCUDAStream()>>>(
            bs*ch,
            input_ptr,
            guide_ptr,
            output_ptr,
            max_ptr,
            dominant_ptr,
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