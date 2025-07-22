#include <cuda_fp16.h>
/************************************************/
/*****************3D modules*********************/
/************************************************/
__device__ void Proj_funcPD3D_iso(float *P1, float *P2, float *P3)
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

__device__ void Proj_funcPD3D_aniso(float *P1, float *P2, float *P3)
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

__device__ void dualPD3D(float *U, float *P1, float *P2, float *P3, float sigma, int methodTV)
{
  *P1 += sigma * (U[1] - U[0]);
  *P2 += sigma * (U[2] - U[0]);
  *P3 += sigma * (U[3] - U[0]);

  if (methodTV == 0)
  {
    Proj_funcPD3D_iso(P1, P2, P3);
  }
  else
  {
    Proj_funcPD3D_aniso(P1, P2, P3);
  }
}

__device__ float DivProj3D(float Input, float U_in, float P1, float P2, float P3, float P1_prev_x, float P2_prev_y, float P3_prev_z, float tau, float lt)
{
  float P_v1 = -(P1 - P1_prev_x);
  float P_v2 = -(P2 - P2_prev_y);
  float P_v3 = -(P3 - P3_prev_z);
  float div_var = P_v1 + P_v2 + P_v3;
  return (U_in - tau * div_var + lt * Input) / (1.0f + lt);
}

template <int items_per_thread>
__global__ void primal_dual_for_total_variation_3D(float *Input, float *U_in, float *U_out, __half *P1_in, __half *P2_in, __half *P3_in, __half *P1_out, __half *P2_out, __half *P3_out, float sigma, float tau, float lt, float theta, int dimX, int dimY, int dimZ, int nonneg, int methodTV)
{
  // calculate each thread global index
  const long base_xIndex = blockIdx.x * blockDim.x + threadIdx.x * items_per_thread;
  const long yIndex = blockIdx.y * blockDim.y + threadIdx.y;
  const long zIndex = blockIdx.z * blockDim.z + threadIdx.z;

  float Input_value[items_per_thread];
  float U[items_per_thread];
  float P1[items_per_thread];
  float P2[items_per_thread];
  float P3[items_per_thread];

  float P1_prev_x[items_per_thread];
  float P1_prev_y[items_per_thread];
  float P1_prev_z[items_per_thread];

  float P2_prev_x[items_per_thread];
  float P2_prev_y[items_per_thread];
  float P2_prev_z[items_per_thread];

  float P3_prev_x[items_per_thread];
  float P3_prev_y[items_per_thread];
  float P3_prev_z[items_per_thread];

  float U_values[4 * items_per_thread];
  float U_values_prev_x[4 * items_per_thread];
  float U_values_prev_y[4 * items_per_thread];
  float U_values_prev_z[4 * items_per_thread];

  for (int i = 0; i < items_per_thread; i++)
  {
    long xIndex = base_xIndex + i;
    if (xIndex >= dimX || yIndex >= dimY || zIndex >= dimZ)
    {
      return;
    }

    long long xStride = 1;
    long long yStride = dimX;
    long long zStride = dimX * dimY;

    long long index = static_cast<long long>(xIndex) + yStride * static_cast<long long>(yIndex) + zStride * static_cast<long long>(zIndex);
    long long index_prev_x = index - xStride;
    long long index_prev_y = index - yStride;
    long long index_prev_z = index - zStride;

    P1_prev_x[i] = 0.0f;
    P2_prev_x[i] = 0.0f;
    P3_prev_x[i] = 0.0f;
    float U_prev_x = 0.0f;

    P1_prev_y[i] = 0.0f;
    P2_prev_y[i] = 0.0f;
    P3_prev_y[i] = 0.0f;
    float U_prev_y = 0.0f;

    P1_prev_z[i] = 0.0f;
    P2_prev_z[i] = 0.0f;
    P3_prev_z[i] = 0.0f;
    float U_prev_z = 0.0f;

    float U_prev_x_prev_y = 0.0;
    float U_prev_x_prev_z = 0.0f;
    float U_prev_y_prev_z = 0.0f;

    P1[i] = __half2float(P1_in[index]);
    P2[i] = __half2float(P2_in[index]);
    P3[i] = __half2float(P3_in[index]);
    U[i] = U_in[index];
    Input_value[i] = Input[index];

    if (xIndex > 0)
    {
      P1_prev_x[i] = __half2float(P1_in[index_prev_x]);
      P2_prev_x[i] = __half2float(P2_in[index_prev_x]);
      P3_prev_x[i] = __half2float(P3_in[index_prev_x]);
      U_prev_x = U_in[index_prev_x];
    }

    if (yIndex > 0)
    {
      P1_prev_y[i] = __half2float(P1_in[index_prev_y]);
      P2_prev_y[i] = __half2float(P2_in[index_prev_y]);
      P3_prev_y[i] = __half2float(P3_in[index_prev_y]);
      U_prev_y = U_in[index_prev_y];
    }

    if (zIndex > 0)
    {
      P1_prev_z[i] = __half2float(P1_in[index_prev_z]);
      P2_prev_z[i] = __half2float(P2_in[index_prev_z]);
      P3_prev_z[i] = __half2float(P3_in[index_prev_z]);
      U_prev_z = U_in[index_prev_z];
    }

    bool last_x = xIndex == dimX - 1;
    bool last_y = yIndex == dimY - 1;
    bool last_z = zIndex == dimZ - 1;

    if (((xIndex > 0) && last_y) || ((yIndex > 0) && last_x))
    {
      U_prev_x_prev_y = U_in[index - xStride - yStride];
    }

    if (((xIndex > 0) && last_z) || ((zIndex > 0) && last_x))
    {
      U_prev_x_prev_z = U_in[index - xStride - zStride];
    }

    if (((yIndex > 0) && last_z) || ((zIndex > 0) && last_y))
    {
      U_prev_y_prev_z = U_in[index - yStride - zStride];
    }

    U_values[0 + 4 * i] = U[i];
    U_values[1 + 4 * i] = last_x ? U_prev_x : U_in[index + xStride];
    U_values[2 + 4 * i] = last_y ? U_prev_y : U_in[index + yStride];
    U_values[3 + 4 * i] = last_z ? U_prev_z : U_in[index + zStride];

    if (xIndex > 0)
    {
      U_values_prev_x[0 + 4 * i] = U_prev_x;
      U_values_prev_x[1 + 4 * i] = U[i];
      U_values_prev_x[2 + 4 * i] = last_y ? U_prev_x_prev_y : U_in[index - xStride + yStride];
      U_values_prev_x[3 + 4 * i] = last_z ? U_prev_x_prev_z : U_in[index - xStride + zStride];
    }

    if (yIndex > 0)
    {
      U_values_prev_y[0 + 4 * i] = U_prev_y;
      U_values_prev_y[1 + 4 * i] = last_x ? U_prev_x_prev_y : U_in[index + xStride - yStride];
      U_values_prev_y[2 + 4 * i] = U[i];
      U_values_prev_y[3 + 4 * i] = last_z ? U_prev_y_prev_z : U_in[index - yStride + zStride];
    }

    if (zIndex > 0)
    {
      U_values_prev_z[0 + 4 * i] = U_prev_z;
      U_values_prev_z[1 + 4 * i] = last_x ? U_prev_x_prev_z : U_in[index + xStride - zStride];
      U_values_prev_z[2 + 4 * i] = last_y ? U_prev_y_prev_z : U_in[index + yStride - zStride];
      U_values_prev_z[3 + 4 * i] = U[i];
    }
  }

  float new_U[items_per_thread];
  for (int i = 0; i < items_per_thread; i++)
  {
    long xIndex = base_xIndex + i;
    if (xIndex >= dimX || yIndex >= dimY || zIndex >= dimZ)
    {
      return;
    }

    dualPD3D(&U_values[4 * i], &P1[i], &P2[i], &P3[i], sigma, methodTV);

    if (xIndex > 0)
    {
      dualPD3D(&U_values_prev_x[4 * i], &P1_prev_x[i], &P2_prev_x[i], &P3_prev_x[i], sigma, methodTV);
    }

    if (yIndex > 0)
    {
      dualPD3D(&U_values_prev_y[4 * i], &P1_prev_y[i], &P2_prev_y[i], &P3_prev_y[i], sigma, methodTV);
    }

    if (zIndex > 0)
    {
      dualPD3D(&U_values_prev_z[4 * i], &P1_prev_z[i], &P2_prev_z[i], &P3_prev_z[i], sigma, methodTV);
    }

    if (nonneg != 0 && U[i] < 0.0f)
    {
      U[i] = 0.0f;
    }

    new_U[i] = DivProj3D(Input_value[i], U[i], P1[i], P2[i], P3[i], P1_prev_x[i], P2_prev_y[i], P3_prev_z[i], tau, lt);
    new_U[i] = new_U[i] + theta * (new_U[i] - U[i]);
  }

  for (int i = 0; i < items_per_thread; i++)
  {
    long xIndex = base_xIndex + i;
    if (xIndex >= dimX || yIndex >= dimY || zIndex >= dimZ)
    {
      return;
    }

    long long index = static_cast<long long>(xIndex) + dimX * static_cast<long long>(yIndex) + dimX * dimY * static_cast<long long>(zIndex);

    U_out[index] = new_U[i];

    P1_out[index] = __float2half(P1[i]);
    P2_out[index] = __float2half(P2[i]);
    P3_out[index] = __float2half(P3[i]);
  }
}

/************************************************/
/*****************2D modules*********************/
/************************************************/
__device__ float2 dualPD(float *U, float sigma, int N, int M, int xIndex, int yIndex, int index)
{
  float P1 = 0.0f;
  float P2 = 0.0f;

  if (xIndex == N - 1)
    P1 += sigma * (U[(xIndex - 1) + N * yIndex] - U[index]);
  else
    P1 += sigma * (U[(xIndex + 1) + N * yIndex] - U[index]);

  if (yIndex == M - 1)
    P2 += sigma * (U[xIndex + N * (yIndex - 1)] - U[index]);
  else
    P2 += sigma * (U[xIndex + N * (yIndex + 1)] - U[index]);

  return make_float2(P1, P2);
}

extern "C" __global__ void primal_dual_for_total_variation_2D(float *U, float sigma, int N, int M, bool nonneg)
{
  // calculate each thread global index
  const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

  if (xIndex >= N || yIndex >= M)
  {
    return;
  }

  int index = xIndex + N * yIndex;
  float2 P1_P2 = dualPD(U, sigma, N, M, xIndex, yIndex, index);
}
