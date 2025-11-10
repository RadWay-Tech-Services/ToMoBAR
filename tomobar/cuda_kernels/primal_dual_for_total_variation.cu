#include <cuda_fp16.h>

/*
Raw CUDA Kernels for Primal-Dual Total Variation regularisation model
*/

template <bool nonneg>
__device__ float clamp_to_zero(float value)
{
}

template <>
__device__ float clamp_to_zero<false>(float value)
{
  return value;
}

template <>
__device__ float clamp_to_zero<true>(float value)
{
  return value < 0.0f ? 0.0f : value;
}

template <typename T>
__device__ float read_as_float(T *P, int index)
{
}

template <>
__device__ float read_as_float<float>(float *P, int index)
{
  return P[index];
}

template <>
__device__ float read_as_float<__half>(__half *P, int index)
{
  return __half2float(P[index]);
}

template <typename T>
__device__ void write_float(T *P, int index, float value)
{
}

template <>
__device__ void write_float<float>(float *P, int index, float value)
{
  P[index] = value;
}

template <>
__device__ void write_float<__half>(__half *P, int index, float value)
{
  P[index] = __float2half(value);
}

/************************************************/
/*****************3D modules*********************/
/************************************************/
template <bool methodTV>
__device__ void Proj_funcPD3D(float *P1, float *P2, float *P3)
{
}

template <>
__device__ void Proj_funcPD3D<false>(float *P1, float *P2, float *P3)
{
  float denom = *P1 * *P1 + *P2 * *P2 + *P3 * *P3;
  if (denom > 1.0f)
  {
    float sq_denom = 1.0f / sqrtf(denom);
    *P1 *= sq_denom;
    *P2 *= sq_denom;
    *P3 *= sq_denom;
  }
}

template <>
__device__ void Proj_funcPD3D<true>(float *P1, float *P2, float *P3)
{
  float val1 = fabs(*P1);
  float val2 = fabs(*P2);
  float val3 = fabs(*P3);

  if (val1 < 1.0f)
  {
    val1 = 1.0f;
  }

  if (val2 < 1.0f)
  {
    val2 = 1.0f;
  }

  if (val3 < 1.0f)
  {
    val3 = 1.0f;
  }

  *P1 /= val1;
  *P2 /= val2;
  *P3 /= val3;
}

template <bool methodTV>
__device__ void dualPD3D(float *U, float *P1, float *P2, float *P3, float sigma)
{
  *P1 += sigma * (U[1] - U[0]);
  *P2 += sigma * (U[2] - U[0]);
  *P3 += sigma * (U[3] - U[0]);

  Proj_funcPD3D<methodTV>(P1, P2, P3);
}

__device__ float DivProj3D(float Input, float U_in, float P1, float P2, float P3, float P1_prev_x, float P2_prev_y, float P3_prev_z, float tau, float lt)
{
  float P_v1 = -(P1 - P1_prev_x);
  float P_v2 = -(P2 - P2_prev_y);
  float P_v3 = -(P3 - P3_prev_z);
  float div_var = P_v1 + P_v2 + P_v3;
  return (U_in - tau * div_var + lt * Input) / (1.0f + lt);
}

__device__ __forceinline__ int calculate_index(int i, int j, int k, int dimX, int dimY, int dimZ)
{
    return static_cast<int>(i) + dimX * static_cast<int>(j) + dimX * dimY * static_cast<int>(k);
}

template<typename T>
__device__ void padded_read_to_shared(T* input, float* shared_out, int dimX, int dimY, int dimZ)
{
  constexpr int padding = 2;
  constexpr int half_padding = padding / 2;
  uint3 sharedDim = make_uint3(blockDim.x + padding, blockDim.y + padding, blockDim.z + padding);
  int shared_memory_size = sharedDim.x * sharedDim.y * sharedDim.z;

  int i_start = blockDim.x * blockIdx.x;
  int j_start = blockDim.y * blockIdx.y;
  int k_start = blockDim.z * blockIdx.z;

  int linear_thread_index = calculate_index(threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockDim.y, blockDim.z);
  int threads_in_block = blockDim.x * blockDim.y * blockDim.z;

  for (int linear_index = linear_thread_index; linear_index < shared_memory_size; linear_index += threads_in_block)
  {
      int tmp = linear_index;
      int shared_x = tmp % sharedDim.x;
      tmp /= sharedDim.x;
      int shared_y = tmp % sharedDim.y;
      tmp /= sharedDim.y;
      int shared_z = tmp;

      int i = i_start - half_padding + shared_x;
      int j = j_start - half_padding + shared_y;
      int k = k_start - half_padding + shared_z;

      if (i < 0 || i >= dimX || j < 0 || j >= dimY || k < 0 || k >= dimZ)
      {
        shared_out[linear_index] = 0.F;
      }
      else
      {
        shared_out[linear_index] = read_as_float<T>(input, calculate_index(i, j, k, dimX, dimY, dimZ));
      }
  }
}

__device__ __forceinline__ int calculate_block_local_index(int i, int j, int k)
{
  constexpr int padding = 2;
  constexpr int half_padding = padding / 2;
  uint3 sharedDim = make_uint3(blockDim.x + padding, blockDim.y + padding, blockDim.z + padding);
  return static_cast<int>(i + half_padding) + sharedDim.x * static_cast<int>(j + half_padding) + sharedDim.x * sharedDim.y * static_cast<int>(k + half_padding);
}

extern __shared__ float U_shared[];

template <typename T, bool nonneg, bool methodTV>
__global__ void primal_dual_for_total_variation_3D(float *Input, float *U_in, float *U_out, T *P1_in, T *P2_in, T *P3_in, T *P1_out, T *P2_out, T *P3_out, float sigma, float tau, float lt, float theta, int dimX, int dimY, int dimZ)
{
  // calculate each thread global index
  const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
  const int zIndex = blockIdx.z * blockDim.z + threadIdx.z;

  padded_read_to_shared(U_in, U_shared, dimX, dimY, dimZ);
  __syncthreads();

  if (xIndex >= dimX || yIndex >= dimY || zIndex >= dimZ)
  {
    return;
  }

  int xStride = 1;
  int yStride = dimX;
  int zStride = dimX * dimY;

  int index = static_cast<int>(xIndex) + yStride * static_cast<int>(yIndex) + zStride * static_cast<int>(zIndex);
  int index_prev_x = index - xStride;
  int index_prev_y = index - yStride;
  int index_prev_z = index - zStride;

  float P1_prev_x = 0.0f;
  float P2_prev_x = 0.0f;
  float P3_prev_x = 0.0f;

  float P1_prev_y = 0.0f;
  float P2_prev_y = 0.0f;
  float P3_prev_y = 0.0f;

  float P1_prev_z = 0.0f;
  float P2_prev_z = 0.0f;
  float P3_prev_z = 0.0f;

  float P1 = read_as_float<T>(P1_in, index);
  float P2 = read_as_float<T>(P2_in, index);
  float P3 = read_as_float<T>(P3_in, index);
  float U = U_shared[calculate_block_local_index(threadIdx.x, threadIdx.y, threadIdx.z)];
  float Input_value = Input[index];

  if (xIndex > 0)
  {
    P1_prev_x = read_as_float<T>(P1_in, index_prev_x);
    P2_prev_x = read_as_float<T>(P2_in, index_prev_x);
    P3_prev_x = read_as_float<T>(P3_in, index_prev_x);
    // U_prev_x = U_in[index_prev_x];
  }

  if (yIndex > 0)
  {
    P1_prev_y = read_as_float<T>(P1_in, index_prev_y);
    P2_prev_y = read_as_float<T>(P2_in, index_prev_y);
    P3_prev_y = read_as_float<T>(P3_in, index_prev_y);
    // U_prev_y = U_in[index_prev_y];
  }

  if (zIndex > 0)
  {
    P1_prev_z = read_as_float<T>(P1_in, index_prev_z);
    P2_prev_z = read_as_float<T>(P2_in, index_prev_z);
    P3_prev_z = read_as_float<T>(P3_in, index_prev_z);
    // U_prev_z = U_in[index_prev_z];
  }

  bool last_x = xIndex == dimX - 1;
  bool last_y = yIndex == dimY - 1;
  bool last_z = zIndex == dimZ - 1;

  // if (((xIndex > 0) && last_y) || ((yIndex > 0) && last_x))
  // {
  //   U_prev_x_prev_y = U_in[index - xStride - yStride];
  // }

  // if (((xIndex > 0) && last_z) || ((zIndex > 0) && last_x))
  // {
  //   U_prev_x_prev_z = U_in[index - xStride - zStride];
  // }

  // if (((yIndex > 0) && last_z) || ((zIndex > 0) && last_y))
  // {
  //   U_prev_y_prev_z = U_in[index - yStride - zStride];
  // }

  {
    float U_values[4] = {
        U,
        last_x ? U_shared[calculate_block_local_index(threadIdx.x - 1, threadIdx.y, threadIdx.z)] : U_shared[calculate_block_local_index(threadIdx.x + 1, threadIdx.y, threadIdx.z)],
        last_y ? U_shared[calculate_block_local_index(threadIdx.x, threadIdx.y - 1, threadIdx.z)] : U_shared[calculate_block_local_index(threadIdx.x, threadIdx.y + 1, threadIdx.z)],
        last_z ? U_shared[calculate_block_local_index(threadIdx.x, threadIdx.y, threadIdx.z - 1)] : U_shared[calculate_block_local_index(threadIdx.x, threadIdx.y, threadIdx.z + 1)]};
    dualPD3D<methodTV>(U_values, &P1, &P2, &P3, sigma);
  }

  if (xIndex > 0)
  {
    float U_values[4] = {
        U_shared[calculate_block_local_index(threadIdx.x - 1, threadIdx.y, threadIdx.z)],
        U,
        last_y ? U_shared[calculate_block_local_index(threadIdx.x - 1, threadIdx.y - 1, threadIdx.z)] : U_shared[calculate_block_local_index(threadIdx.x - 1, threadIdx.y + 1, threadIdx.z)],
        last_z ? U_shared[calculate_block_local_index(threadIdx.x - 1, threadIdx.y, threadIdx.z - 1)] : U_shared[calculate_block_local_index(threadIdx.x - 1, threadIdx.y, threadIdx.z + 1)]};
    dualPD3D<methodTV>(U_values, &P1_prev_x, &P2_prev_x, &P3_prev_x, sigma);
  }

  if (yIndex > 0)
  {
    float U_values[4] = {
        U_shared[calculate_block_local_index(threadIdx.x, threadIdx.y - 1, threadIdx.z)],
        last_x ? U_shared[calculate_block_local_index(threadIdx.x - 1, threadIdx.y - 1, threadIdx.z)] : U_shared[calculate_block_local_index(threadIdx.x + 1, threadIdx.y - 1, threadIdx.z)],
        U,
        last_z ? U_shared[calculate_block_local_index(threadIdx.x, threadIdx.y - 1, threadIdx.z - 1)] : U_shared[calculate_block_local_index(threadIdx.x, threadIdx.y - 1, threadIdx.z + 1)]};
    dualPD3D<methodTV>(U_values, &P1_prev_y, &P2_prev_y, &P3_prev_y, sigma);
  }

  if (zIndex > 0)
  {
    float U_values[4] = {
        U_shared[calculate_block_local_index(threadIdx.x, threadIdx.y, threadIdx.z - 1)],
        last_x ? U_shared[calculate_block_local_index(threadIdx.x - 1, threadIdx.y, threadIdx.z - 1)] : U_shared[calculate_block_local_index(threadIdx.x + 1, threadIdx.y, threadIdx.z - 1)],
        last_y ? U_shared[calculate_block_local_index(threadIdx.x, threadIdx.y - 1, threadIdx.z - 1)] : U_shared[calculate_block_local_index(threadIdx.x, threadIdx.y + 1, threadIdx.z - 1)],
        U};
    dualPD3D<methodTV>(U_values, &P1_prev_z, &P2_prev_z, &P3_prev_z, sigma);
  }

  U = clamp_to_zero<nonneg>(U);
  float new_U = DivProj3D(Input_value, U, P1, P2, P3, P1_prev_x, P2_prev_y, P3_prev_z, tau, lt);
  U_out[index] = new_U + theta * (new_U - U);

  write_float<T>(P1_out, index, P1);
  write_float<T>(P2_out, index, P2);
  write_float<T>(P3_out, index, P3);
}

/************************************************/
/*****************2D modules*********************/
/************************************************/
template <bool methodTV>
__device__ void Proj_funcPD2D(float *P1, float *P2)
{
}

template <>
__device__ void Proj_funcPD2D<false>(float *P1, float *P2)
{
  float denom = *P1 * *P1 + *P2 * *P2;
  if (denom > 1.0f)
  {
    float sq_denom = 1.0f / sqrtf(denom);
    *P1 *= sq_denom;
    *P2 *= sq_denom;
  }
}

template <>
__device__ void Proj_funcPD2D<true>(float *P1, float *P2)
{
  float val1 = fabs(*P1);
  float val2 = fabs(*P2);

  if (val1 < 1.0f)
  {
    val1 = 1.0f;
  }

  if (val2 < 1.0f)
  {
    val2 = 1.0f;
  }

  *P1 /= val1;
  *P2 /= val2;
}

template <bool methodTV>
__device__ void dualPD2D(float *U, float *P1, float *P2, float sigma)
{
  *P1 += sigma * (U[1] - U[0]);
  *P2 += sigma * (U[2] - U[0]);

  Proj_funcPD2D<methodTV>(P1, P2);
}

__device__ float DivProj2D(float Input, float U_in, float P1, float P2, float P1_prev_x, float P2_prev_y, float tau, float lt)
{
  float P_v1 = -(P1 - P1_prev_x);
  float P_v2 = -(P2 - P2_prev_y);
  float div_var = P_v1 + P_v2;
  return (U_in - tau * div_var + lt * Input) / (1.0f + lt);
}

template <typename T, bool nonneg, bool methodTV>
__global__ void primal_dual_for_total_variation_2D(float *Input, float *U_in, float *U_out, T *P1_in, T *P2_in, T *P1_out, T *P2_out, float sigma, float tau, float lt, float theta, int dimX, int dimY)
{
  // calculate each thread global index
  const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

  if (xIndex >= dimX || yIndex >= dimY)
  {
    return;
  }

  int xStride = 1;
  int yStride = dimX;

  int index = static_cast<int>(xIndex) + yStride * static_cast<int>(yIndex);
  int index_prev_x = index - xStride;
  int index_prev_y = index - yStride;

  float P1_prev_x = 0.0f;
  float P2_prev_x = 0.0f;
  float U_prev_x = 0.0f;

  float P1_prev_y = 0.0f;
  float P2_prev_y = 0.0f;
  float U_prev_y = 0.0f;

  float U_prev_x_prev_y = 0.0;

  float P1 = read_as_float<T>(P1_in, index);
  float P2 = read_as_float<T>(P2_in, index);
  float U = U_in[index];
  float Input_value = Input[index];

  if (xIndex > 0)
  {
    P1_prev_x = read_as_float<T>(P1_in, index_prev_x);
    P2_prev_x = read_as_float<T>(P2_in, index_prev_x);
    U_prev_x = U_in[index_prev_x];
  }

  if (yIndex > 0)
  {
    P1_prev_y = read_as_float<T>(P1_in, index_prev_y);
    P2_prev_y = read_as_float<T>(P2_in, index_prev_y);
    U_prev_y = U_in[index_prev_y];
  }

  bool last_x = xIndex == dimX - 1;
  bool last_y = yIndex == dimY - 1;

  if (((xIndex > 0) && last_y) || ((yIndex > 0) && last_x))
  {
    U_prev_x_prev_y = U_in[index - xStride - yStride];
  }

  {
    float U_values[3] = {
        U,
        last_x ? U_prev_x : U_in[index + xStride],
        last_y ? U_prev_y : U_in[index + yStride],
    };

    dualPD2D<methodTV>(U_values, &P1, &P2, sigma);
  }

  if (xIndex > 0)
  {
    float U_values[3] = {
        U_prev_x,
        U,
        last_y ? U_prev_x_prev_y : U_in[index - xStride + yStride],
    };
    dualPD2D<methodTV>(U_values, &P1_prev_x, &P2_prev_x, sigma);
  }

  if (yIndex > 0)
  {
    float U_values[3] = {
        U_prev_y,
        last_x ? U_prev_x_prev_y : U_in[index + xStride - yStride],
        U,
    };
    dualPD2D<methodTV>(U_values, &P1_prev_y, &P2_prev_y, sigma);
  }

  U = clamp_to_zero<nonneg>(U);
  float new_U = DivProj2D(Input_value, U, P1, P2, P1_prev_x, P2_prev_y, tau, lt);
  U_out[index] = new_U + theta * (new_U - U);

  write_float<T>(P1_out, index, P1);
  write_float<T>(P2_out, index, P2);
}