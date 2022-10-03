// #include <torch/torch.h>
// name should be different from .cpp file!!!
#include <torch/types.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>

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

// The guide cannot be derived by setting 0 and 1
// We set guide as the value of norm at each position across channels
// Then expand it for D channels
// Calculate sigmoid of three pixels to have weighted sum for the current pixel using dynamic programming
//
template <typename scalar_t>
__global__ void maxlen1_forward_kernel(const int nthreads, 
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
            for (int j = 0; j < sw; j++) {
                auto var_current = *(input_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto var_upper = scalar_t(0);
                auto var_left = scalar_t(0);

                auto guide_upper = scalar_t(0);
                auto guide_lefter = scalar_t(0);

                if (i>0) {
                  guide_upper = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j);
                  var_upper = *(outptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j);
                }

                if (j>0) {
                  guide_lefter = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + (j-1));
                  var_left = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + (j-1));
                }
                
                // guide shows the max_len position with 1 while 0 for other positions
                auto guide_current = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);

                auto denominator = exp(guide_current) + exp(guide_lefter) + exp(guide_upper);
                auto upper_weight = exp(guide_upper)/denominator;
                auto lefter_weight = exp(guide_lefter)/denominator;
                auto current_weight = exp(guide_current)/denominator;
                // printf("i:%d,j:%d,sh:%d,sw:%d ",i,j,sh,sw);
                // printf(" vup:%f,vleft:%f,vcurrent:%f,gupper:%f,glefter:%f,gcurrent:%f ",
                //       var_upper,var_left,var_current,guide_upper,guide_lefter,guide_current);
                
                // printf("output:%f\n",current_weight*var_current + upper_weight*var_upper + lefter_weight*var_left);


                outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] =  current_weight*var_current + upper_weight*var_upper + lefter_weight*var_left;
                max_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = current_weight*var_current + upper_weight*var_upper + lefter_weight*var_left;
            }
        }
    }   
}

template <typename scalar_t>
__global__ void maxlen1_backward_kernel(const int nthreads, 
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

                // guide value for upper, lefter, current position
                auto out_current = *(input_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto out_upper = out_current;
                auto out_lefter = out_current;
                // upper,lefter,current value in feature map
                auto current_guider = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto lefter_guider = current_guider;
                auto upper_guider = current_guider;
                
                if (j > 0)  {
                  out_lefter = *(output_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j-1);
                  lefter_guider = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j-1);
                }
                if (i > 0) {
                  out_upper = *(output_ptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j);
                  upper_guider = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j);
                }
                // gradient for guider
                // d_A_B: partial derivative for A on B
                // out_current partial current_guider_weight
                auto d_Cout_Cgweight = out_current;
                auto denominator = pow(exp(lefter_guider)+ exp(upper_guider)+ exp(current_guider),-2);
                auto d_Cgweight_Cg = denominator*(exp(lefter_guider)+ exp(upper_guider))*exp(current_guider);
                auto Cg_grad = d_Cout_Cgweight*d_Cgweight_Cg;

                gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
                  gradout_ptr[db * ch * sh * sw + dc * sh* sw + i * sw + j]*Cg_grad;

                // gradient for grad_input
                auto Cgweight = exp(current_guider)/(exp(current_guider)+exp(lefter_guider)+exp(upper_guider));
                auto d_Cout_Cinput = Cgweight;

                // backforward gradient flow for left and upper position
                auto Lgweight = exp(lefter_guider)/(exp(current_guider)+exp(lefter_guider)+exp(upper_guider));
                auto d_Cout_Linput = Lgweight;

                auto Ugweight = exp(upper_guider)/(exp(current_guider)+exp(lefter_guider)+exp(upper_guider));
                auto d_Cout_Uinput = Ugweight;

                gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
                  gradout_ptr[db * ch * sh * sw + dc * sh* sw + i* sw + j]*d_Cout_Cinput;
                
                // backward gradient flow
                if (j > 0) gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j-1] += 
                  d_Cout_Linput* gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];


                if (i > 0) gradout_ptr[db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j] += 
                  d_Cout_Uinput* gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];

            }
        }
    }   
}


template <typename scalar_t>
__global__ void maxlen2_forward_kernel(const int nthreads, 
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

                auto var_current = *(input_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto var_upper = scalar_t(0);
                auto var_right = scalar_t(0);

                auto guide_upper = scalar_t(0);
                auto guide_righter = scalar_t(0);

                if (i>0) {
                  guide_upper = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j);
                  var_upper = *(outptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j);
                }

                if (j < sw-1) {
                  guide_righter = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + (j+1));
                  var_right = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + (j+1));
                }
                
                // guide shows the max_len position with 1 while 0 for other positions
                auto guide_current = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);

                auto denominator = exp(guide_current) + exp(guide_righter) + exp(guide_upper);
                auto upper_weight = exp(guide_upper)/denominator;
                auto righter_weight = exp(guide_righter)/denominator;
                auto current_weight = exp(guide_current)/denominator;
                // printf("i:%d,j:%d,sh:%d,sw:%d ",i,j,sh,sw);
                // printf(" vup:%f,vleft:%f,vcurrent:%f,gupper:%f,glefter:%f,gcurrent:%f ",
                //       var_upper,var_left,var_current,guide_upper,guide_lefter,guide_current);
                
                // printf("output:%f\n",current_weight*var_current + upper_weight*var_upper + lefter_weight*var_left);


                outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] =  current_weight*var_current + upper_weight*var_upper + righter_weight*var_right;
                max_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = current_weight*var_current + upper_weight*var_upper + righter_weight*var_right;
            
            }
        }
        
    }   
}

template <typename scalar_t>
__global__ void maxlen2_backward_kernel(const int nthreads, 
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

                // guide value for upper, lefter, current position
                auto out_current = *(input_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto out_upper = out_current;
                auto out_righter = out_current;
                // upper,lefter,current value in feature map
                auto current_guider = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto righter_guider = current_guider;
                auto upper_guider = current_guider;
                
                if (j < sw-1)  {
                  out_righter = *(output_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j+1);
                  righter_guider = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j+1);
                }
                if (i > 0) {
                  out_upper = *(output_ptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j);
                  upper_guider = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j);
                }
                // gradient for guider
                // d_A_B: partial derivative for A on B
                // out_current partial current_guider_weight
                auto d_Cout_Cgweight = out_current;
                auto denominator = pow(exp(righter_guider)+ exp(upper_guider)+ exp(current_guider),-2);
                auto d_Cgweight_Cg = denominator*(exp(righter_guider)+ exp(upper_guider))*exp(current_guider);
                auto Cg_grad = d_Cout_Cgweight*d_Cgweight_Cg;

                gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
                  gradout_ptr[db * ch * sh * sw + dc * sh* sw + i * sw + j]*Cg_grad;

                // gradient for grad_input
                auto Cgweight = exp(current_guider)/(exp(current_guider)+exp(righter_guider)+exp(upper_guider));
                auto d_Cout_Cinput = Cgweight;

                // backforward gradient flow for left and upper position
                auto Rgweight = exp(righter_guider)/(exp(current_guider)+exp(righter_guider)+exp(upper_guider));
                auto d_Cout_Linput = Rgweight;

                auto Ugweight = exp(upper_guider)/(exp(current_guider)+exp(righter_guider)+exp(upper_guider));
                auto d_Cout_Uinput = Ugweight;

                gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
                  gradout_ptr[db * ch * sh * sw + dc * sh* sw + i* sw + j]*d_Cout_Cinput;
                
                // backward gradient flow
                if (j < sw-1) gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j+1] += 
                  d_Cout_Linput* gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];


                if (i > 0) gradout_ptr[db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j] += 
                  d_Cout_Uinput* gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];

                
            }
        }
        
    }   
}

template <typename scalar_t>
__global__ void maxlen3_forward_kernel(const int nthreads, 
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
                auto var_current = *(input_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto var_downer = scalar_t(0);
                auto var_left = scalar_t(0);

                auto guide_downer = scalar_t(0);
                auto guide_lefter = scalar_t(0);

                if (i < sh-1) {
                  guide_downer = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j);
                  var_downer = *(outptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j);
                }

                if (j>0) {
                  guide_lefter = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + (j-1));
                  var_left = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + (j-1));
                }
                
                // guide shows the max_len position with 1 while 0 for other positions
                auto guide_current = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);

                auto denominator = exp(guide_current) + exp(guide_lefter) + exp(guide_downer);
                auto downer_weight = exp(guide_downer)/denominator;
                auto lefter_weight = exp(guide_lefter)/denominator;
                auto current_weight = exp(guide_current)/denominator;
                // printf("i:%d,j:%d,sh:%d,sw:%d ",i,j,sh,sw);
                // printf(" vup:%f,vleft:%f,vcurrent:%f,gupper:%f,glefter:%f,gcurrent:%f ",
                //       var_upper,var_left,var_current,guide_upper,guide_lefter,guide_current);
                
                // printf("output:%f\n",current_weight*var_current + upper_weight*var_upper + lefter_weight*var_left);

                outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] =  current_weight*var_current + downer_weight*var_downer + lefter_weight*var_left;
                max_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = current_weight*var_current + downer_weight*var_downer + lefter_weight*var_left;

                

            }
        }
        
    }   
}

template <typename scalar_t>
__global__ void maxlen3_backward_kernel(const int nthreads, 
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

                // guide value for upper, lefter, current position
                auto out_current = *(input_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto out_downer = out_current;
                auto out_lefter = out_current;
                // upper,lefter,current value in feature map
                auto current_guider = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto lefter_guider = current_guider;
                auto downer_guider = current_guider;
                
                if (j > 0)  {
                  out_lefter = *(output_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j-1);
                  lefter_guider = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j-1);
                }
                if (i < sh-1) {
                  out_downer = *(output_ptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j);
                  downer_guider = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j);
                }
                // gradient for guider
                // d_A_B: partial derivative for A on B
                // out_current partial current_guider_weight
                auto d_Cout_Cgweight = out_current;
                auto denominator = pow(exp(lefter_guider)+ exp(downer_guider)+ exp(current_guider),-2);
                auto d_Cgweight_Cg = denominator*(exp(lefter_guider)+ exp(downer_guider))*exp(current_guider);
                auto Cg_grad = d_Cout_Cgweight*d_Cgweight_Cg;

                gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
                  gradout_ptr[db * ch * sh * sw + dc * sh* sw + i * sw + j]*Cg_grad;

                // gradient for grad_input
                auto Cgweight = exp(current_guider)/(exp(current_guider)+exp(lefter_guider)+exp(downer_guider));
                auto d_Cout_Cinput = Cgweight;

                // backforward gradient flow for left and upper position
                auto Lgweight = exp(lefter_guider)/(exp(current_guider)+exp(lefter_guider)+exp(downer_guider));
                auto d_Cout_Linput = Lgweight;

                auto Dgweight = exp(downer_guider)/(exp(current_guider)+exp(lefter_guider)+exp(downer_guider));
                auto d_Cout_Uinput = Dgweight;

                gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
                  gradout_ptr[db * ch * sh * sw + dc * sh* sw + i* sw + j]*d_Cout_Cinput;
                
                // backward gradient flow
                if (j > 0) gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j-1] += 
                  d_Cout_Linput* gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];

                if (i < sh-1) gradout_ptr[db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j] += 
                  d_Cout_Uinput* gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];

                
            }
        }
        
    }   
}



template <typename scalar_t>
__global__ void maxlen4_forward_kernel(const int nthreads, 
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
                auto var_current = *(input_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto var_downer = scalar_t(0);
                auto var_right = scalar_t(0);

                auto guide_downer = scalar_t(0);
                auto guide_righter = scalar_t(0);

                if (i < sh-1) {
                  guide_downer = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j);
                  var_downer = *(outptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j);
                }

                if (j < sw-1) {
                  guide_righter = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + (j+1));
                  var_right = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + (j+1));
                }
                
                // guide shows the max_len position with 1 while 0 for other positions
                auto guide_current = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);

                auto denominator = exp(guide_current) + exp(guide_righter) + exp(guide_downer);
                auto downer_weight = exp(guide_downer)/denominator;
                auto righter_weight = exp(guide_righter)/denominator;
                auto current_weight = exp(guide_current)/denominator;
                // printf("i:%d,j:%d,sh:%d,sw:%d ",i,j,sh,sw);
                // printf(" vup:%f,vleft:%f,vcurrent:%f,gupper:%f,glefter:%f,gcurrent:%f ",
                //       var_upper,var_left,var_current,guide_upper,guide_lefter,guide_current);
                
                // printf("output:%f\n",current_weight*var_current + upper_weight*var_upper + lefter_weight*var_left);


                outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] =  current_weight*var_current + downer_weight*var_downer + righter_weight*var_right;
                max_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = current_weight*var_current + downer_weight*var_downer + righter_weight*var_right;
            
            }
        }
        
    }   
}

template <typename scalar_t>
__global__ void maxlen4_backward_kernel(const int nthreads, 
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

                // guide value for upper, lefter, current position
                auto out_current = *(input_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto out_downer = out_current;
                auto out_righter = out_current;
                // upper,lefter,current value in feature map
                auto current_guider = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto righter_guider = current_guider;
                auto downer_guider = current_guider;
                
                if (j < sw-1)  {
                  out_righter = *(output_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j+1);
                  righter_guider = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j+1);
                }
                if (i < sh-1) {
                  out_downer = *(output_ptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j);
                  downer_guider = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j);
                }
                // gradient for guider
                // d_A_B: partial derivative for A on B
                // out_current partial current_guider_weight
                auto d_Cout_Cgweight = out_current;
                auto denominator = pow(exp(righter_guider)+ exp(downer_guider)+ exp(current_guider),-2);
                auto d_Cgweight_Cg = denominator*(exp(righter_guider)+ exp(downer_guider))*exp(current_guider);
                auto Cg_grad = d_Cout_Cgweight*d_Cgweight_Cg;

                gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
                  gradout_ptr[db * ch * sh * sw + dc * sh* sw + i * sw + j]*Cg_grad;

                // gradient for grad_input
                auto Cgweight = exp(current_guider)/(exp(current_guider)+exp(righter_guider)+exp(downer_guider));
                auto d_Cout_Cinput = Cgweight;

                // backforward gradient flow for left and upper position
                auto Rgweight = exp(righter_guider)/(exp(current_guider)+exp(righter_guider)+exp(downer_guider));
                auto d_Cout_Linput = Rgweight;

                auto Ugweight = exp(downer_guider)/(exp(current_guider)+exp(righter_guider)+exp(downer_guider));
                auto d_Cout_Uinput = Ugweight;

                gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
                  gradout_ptr[db * ch * sh * sw + dc * sh* sw + i* sw + j]*d_Cout_Cinput;
                
                // backward gradient flow
                if (j < sw-1) gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j+1] += 
                  d_Cout_Linput* gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];


                if (i < sh-1) gradout_ptr[db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j] += 
                  d_Cout_Uinput* gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];

                
            }
        }
        
    }   
}


namespace landmarkconv {

std::vector<at::Tensor> maxlen1_pool_forward_laucher(
    const at::Tensor &input, 
    const at::Tensor &guide) {
    // Ensure CUDA uses the input tensor device.
    at::DeviceGuard guard(input.device());
    AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");
    // printf("call by cuda...\n");

    cudaDeviceSynchronize(); // for print
    auto output = input.clone();
    auto maxout = input.clone();
    int bs = input.size(0);
    int ch = input.size(1);
    int sh = input.size(2);
    int sw = input.size(3);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "maxlen1_pool_forward_laucher", ([&] {
            const scalar_t *input_ptr = input.data_ptr<scalar_t>();
            const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
            scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
            scalar_t *output_ptr = output.data_ptr<scalar_t>();
            maxlen1_forward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
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


std::vector<at::Tensor> maxlen1_pool_backward_laucher(
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
    input.scalar_type(), "maxlen1_pool_backward_laucher", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
        const scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
        const scalar_t *output_ptr = output.data_ptr<scalar_t>();
        scalar_t *gradout_ptr = gradout.data_ptr<scalar_t>();
        scalar_t *gradin_ptr = grad_input.data_ptr<scalar_t>();
        scalar_t *gradguide_ptr = grad_guide.data_ptr<scalar_t>();


        maxlen1_backward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
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


std::vector<at::Tensor> maxlen2_pool_forward_laucher(
  const at::Tensor &input, 
  const at::Tensor &guide) {
  // Ensure CUDA uses the input tensor device.
  at::DeviceGuard guard(input.device());
  AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");
  // printf("call by cuda...\n");

  cudaDeviceSynchronize(); // for print
  auto output = input.clone();
  auto maxout = input.clone();
  int bs = input.size(0);
  int ch = input.size(1);
  int sh = input.size(2);
  int sw = input.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "maxlen2_pool_forward_laucher", ([&] {
          const scalar_t *input_ptr = input.data_ptr<scalar_t>();
          const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
          scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
          scalar_t *output_ptr = output.data_ptr<scalar_t>();
          maxlen2_forward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
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


std::vector<at::Tensor> maxlen2_pool_backward_laucher(
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
    input.scalar_type(), "maxlen2_pool_backward_laucher", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
        const scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
        const scalar_t *output_ptr = output.data_ptr<scalar_t>();
        scalar_t *gradout_ptr = gradout.data_ptr<scalar_t>();
        scalar_t *gradin_ptr = grad_input.data_ptr<scalar_t>();
        scalar_t *gradguide_ptr = grad_guide.data_ptr<scalar_t>();


        maxlen2_backward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
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


std::vector<at::Tensor> maxlen3_pool_forward_laucher(
  const at::Tensor &input, 
  const at::Tensor &guide) {
  // Ensure CUDA uses the input tensor device.
  at::DeviceGuard guard(input.device());
  AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");
  // printf("call by cuda...\n");

  cudaDeviceSynchronize(); // for print
  auto output = input.clone();
  auto maxout = input.clone();
  int bs = input.size(0);
  int ch = input.size(1);
  int sh = input.size(2);
  int sw = input.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "maxlen3_pool_forward_laucher", ([&] {
          const scalar_t *input_ptr = input.data_ptr<scalar_t>();
          const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
          scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
          scalar_t *output_ptr = output.data_ptr<scalar_t>();
          maxlen3_forward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
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


std::vector<at::Tensor> maxlen3_pool_backward_laucher(
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
    input.scalar_type(), "maxlen3_pool_backward_laucher", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
        const scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
        const scalar_t *output_ptr = output.data_ptr<scalar_t>();
        scalar_t *gradout_ptr = gradout.data_ptr<scalar_t>();
        scalar_t *gradin_ptr = grad_input.data_ptr<scalar_t>();
        scalar_t *gradguide_ptr = grad_guide.data_ptr<scalar_t>();


        maxlen3_backward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
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



std::vector<at::Tensor> maxlen4_pool_forward_laucher(
  const at::Tensor &input, 
  const at::Tensor &guide) {
  // Ensure CUDA uses the input tensor device.
  at::DeviceGuard guard(input.device());
  AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");
  // printf("call by cuda...\n");

  cudaDeviceSynchronize(); // for print
  auto output = input.clone();
  auto maxout = input.clone();
  int bs = input.size(0);
  int ch = input.size(1);
  int sh = input.size(2);
  int sw = input.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "maxlen4_pool_forward_laucher", ([&] {
          const scalar_t *input_ptr = input.data_ptr<scalar_t>();
          const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
          scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
          scalar_t *output_ptr = output.data_ptr<scalar_t>();
          maxlen4_forward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
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


std::vector<at::Tensor> maxlen4_pool_backward_laucher(
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
    input.scalar_type(), "maxlen4_pool_backward_laucher", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
        const scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
        const scalar_t *output_ptr = output.data_ptr<scalar_t>();
        scalar_t *gradout_ptr = gradout.data_ptr<scalar_t>();
        scalar_t *gradin_ptr = grad_input.data_ptr<scalar_t>();
        scalar_t *gradguide_ptr = grad_guide.data_ptr<scalar_t>();


        maxlen4_backward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
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