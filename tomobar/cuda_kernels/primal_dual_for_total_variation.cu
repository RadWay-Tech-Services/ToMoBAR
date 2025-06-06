/************************************************/
/*****************3D modules*********************/
/************************************************/
__device__ float3 Proj_funcPD3D_iso(float P1, float P2, float P3)
{
  float denom = P1 * P1 + P2 * P2 + P3 * P3;
  if (denom > 1.0f)
  {
    float sq_denom = 1.0f / sqrtf(denom);
    P1 *= sq_denom;
    P2 *= sq_denom;
    P3 *= sq_denom;
  }

  return make_float3(P1, P2, P3);
}

__device__ float3 Proj_funcPD3D_aniso(float P1, float P2, float P3)
{
  float val1 = abs(P1);
  float val2 = abs(P2);
  float val3 = abs(P3);

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

  P1 /= val1;
  P2 /= val2;
  P3 /= val3;

  return make_float3(P1, P2, P3);
}

__device__ float3 dualPD3D(float *U, float *P1, float *P2, float *P3, float sigma, int dimX, int dimY, int dimZ, long i, long j, long k, long long index, int methodTV)
{
  float P1_local = P1[index];
  float P2_local = P2[index];
  float P3_local = P3[index];

  if (i == dimX - 1)
  {
    long long index1 = static_cast<long long>(i - 1) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);
    P1_local += sigma * (U[index1] - U[index]);
  }
  else
  {
    long long index2 = static_cast<long long>(i + 1) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);
    P1_local += sigma * (U[index2] - U[index]);
  }

  if (j == dimY - 1)
  {
    long long index3 = static_cast<long long>(i) + dimX * static_cast<long long>(j - 1) + dimX * dimY * static_cast<long long>(k);
    P2_local += sigma * (U[index3] - U[index]);
  }
  else
  {
    long long index4 = static_cast<long long>(i) + dimX * static_cast<long long>(j + 1) + dimX * dimY * static_cast<long long>(k);
    P2_local += sigma * (U[index4] - U[index]);
  }

  if (k == dimZ - 1)
  {
    long long index5 = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k - 1);
    P3_local += sigma * (U[index5] - U[index]);
  }
  else
  {
    long long index6 = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k + 1);
    P3_local += sigma * (U[index6] - U[index]);
  }

  if (methodTV == 0)
  {
    return Proj_funcPD3D_iso(P1_local, P2_local, P3_local);
  }
  else
  {
    return Proj_funcPD3D_aniso(P1_local, P2_local, P3_local);
  }
}

__device__ float DivProj3D(float *Input, float U_in, float3 P1_P2_P3, float3 shifted_P1_P2_P3, float tau, float lt, int dimX, int dimY, long i, long j, long k, long long index)
{
  float P_v1, P_v2, P_v3;

  if (i == 0)
  {
    P_v1 = -P1_P2_P3.x;
  }
  else
  {
    P_v1 = -(P1_P2_P3.x - shifted_P1_P2_P3.x);
  }

  if (j == 0)
  {
    P_v2 = -P1_P2_P3.y;
  }
  else
  {
    P_v2 = -(P1_P2_P3.y - shifted_P1_P2_P3.y);
  }

  if (k == 0)
  {
    P_v3 = -P1_P2_P3.z;
  }
  else
  {
    P_v3 = -(P1_P2_P3.z - shifted_P1_P2_P3.z);
  }

  float div_var = P_v1 + P_v2 + P_v3;

  return (U_in - tau * div_var + lt * Input[index]) / (1.0f + lt);
}

__device__ long long calculate_index(long i, long j, long k, int dimX, int dimY)
{
  return static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);
}

extern "C" __global__ void primal_dual_for_total_variation_3D(float *Input, float *U_in, float *U_out, float *P1_in, float *P2_in, float *P3_in, float *P1_out, float *P2_out, float *P3_out, float sigma, float tau, float lt, float theta, int dimX, int dimY, int dimZ, int nonneg, int methodTV)
{
  // calculate each thread global index
  const long xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const long yIndex = blockIdx.y * blockDim.y + threadIdx.y;
  const long zIndex = blockIdx.z * blockDim.z + threadIdx.z;

  if (xIndex >= dimX || yIndex >= dimY || zIndex >= dimZ)
  {
    return;
  }

  long long index = calculate_index(xIndex, yIndex, zIndex, dimX, dimY);

  float3 P1_P2_P3 = dualPD3D(U_in, P1_in, P2_in, P3_in, sigma, dimX, dimY, dimZ, xIndex, yIndex, zIndex, index, methodTV);
  float3 shifted_P1_P2_P3 = make_float3(
      dualPD3D(U_in, P1_in, P2_in, P3_in, sigma, dimX, dimY, dimZ, xIndex - 1, yIndex, zIndex, calculate_index(xIndex - 1, yIndex, zIndex, dimX, dimY), methodTV).x,
      dualPD3D(U_in, P1_in, P2_in, P3_in, sigma, dimX, dimY, dimZ, xIndex, yIndex - 1, zIndex, calculate_index(xIndex, yIndex - 1, zIndex, dimX, dimY), methodTV).y,
      dualPD3D(U_in, P1_in, P2_in, P3_in, sigma, dimX, dimY, dimZ, xIndex, yIndex, zIndex - 1, calculate_index(xIndex, yIndex, zIndex - 1, dimX, dimY), methodTV).z);

  float old_U = U_in[index];
  if (nonneg != 0 && old_U < 0.0f)
  {
    old_U = 0.0f;
  }

  float new_U = DivProj3D(Input, old_U, P1_P2_P3, shifted_P1_P2_P3, tau, lt, dimX, dimY, xIndex, yIndex, zIndex, index);
  U_out[index] = new_U + theta * (new_U - old_U);

  P1_out[index] = P1_P2_P3.x;
  P2_out[index] = P1_P2_P3.y;
  P3_out[index] = P1_P2_P3.z;
}

extern "C" __global__ void dualPD3D_kernel(float *U, float *P1, float *P2, float *P3, float sigma, int dimX, int dimY, int dimZ)
{

  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  const long j = blockDim.y * blockIdx.y + threadIdx.y;
  const long k = blockDim.z * blockIdx.z + threadIdx.z;

  if (i >= dimX || j >= dimY || k >= dimZ)
    return;

  long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);

  if (i == dimX - 1)
  {
    long long index1 = static_cast<long long>(i - 1) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);
    P1[index] += sigma * (U[index1] - U[index]);
  }
  else
  {
    long long index2 = static_cast<long long>(i + 1) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);
    P1[index] += sigma * (U[index2] - U[index]);
  }
  if (j == dimY - 1)
  {
    long long index3 = static_cast<long long>(i) + dimX * static_cast<long long>(j - 1) + dimX * dimY * static_cast<long long>(k);
    P2[index] += sigma * (U[index3] - U[index]);
  }
  else
  {
    long long index4 = static_cast<long long>(i) + dimX * static_cast<long long>(j + 1) + dimX * dimY * static_cast<long long>(k);
    P2[index] += sigma * (U[index4] - U[index]);
  }
  if (k == dimZ - 1)
  {
    long long index5 = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k - 1);
    P3[index] += sigma * (U[index5] - U[index]);
  }
  else
  {
    long long index6 = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k + 1);
    P3[index] += sigma * (U[index6] - U[index]);
  }

  return;
}
extern "C" __global__ void Proj_funcPD3D_iso_kernel(float *P1, float *P2, float *P3, int dimX, int dimY, int dimZ)
{

  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  const long j = blockDim.y * blockIdx.y + threadIdx.y;
  const long k = blockDim.z * blockIdx.z + threadIdx.z;

  if (i >= dimX || j >= dimY || k >= dimZ)
    return;

  float denom, sq_denom;
  long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);

  denom = P1[index] * P1[index] + P2[index] * P2[index] + P3[index] * P3[index];
  if (denom > 1.0f)
  {
    sq_denom = 1.0f / sqrtf(denom);
    P1[index] *= sq_denom;
    P2[index] *= sq_denom;
    P3[index] *= sq_denom;
  }
  return;
}

extern "C" __global__ void Proj_funcPD3D_aniso_kernel(float *P1, float *P2, float *P3, int dimX, int dimY, int dimZ)
{
  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  const long j = blockDim.y * blockIdx.y + threadIdx.y;
  const long k = blockDim.z * blockIdx.z + threadIdx.z;
  float val1, val2, val3;

  if (i >= dimX || j >= dimY || k >= dimZ)
    return;
  long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);

  val1 = abs(P1[index]);
  val2 = abs(P2[index]);
  val3 = abs(P3[index]);
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
  P1[index] /= val1;
  P2[index] /= val2;
  P3[index] /= val3;

  return;
}
extern "C" __global__ void DivProj3D_kernel(float *U, float *Input, float *P1, float *P2, float *P3, float lt, float tau, int dimX, int dimY, int dimZ)
{
  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  const long j = blockDim.y * blockIdx.y + threadIdx.y;
  const long k = blockDim.z * blockIdx.z + threadIdx.z;
  float P_v1, P_v2, P_v3, div_var;

  if (i >= dimX || j >= dimY || k >= dimZ)
    return;
  long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);

  if (i == 0)
    P_v1 = -P1[index];
  else
  {
    long long index1 = static_cast<long long>(i - 1) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);
    P_v1 = -(P1[index] - P1[index1]);
  }
  if (j == 0)
    P_v2 = -P2[index];
  else
  {
    long long index2 = static_cast<long long>(i) + dimX * static_cast<long long>(j - 1) + dimX * dimY * static_cast<long long>(k);
    P_v2 = -(P2[index] - P2[index2]);
  }
  if (k == 0)
    P_v3 = -P3[index];
  else
  {
    long long index3 = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k - 1);
    P_v3 = -(P3[index] - P3[index3]);
  }

  div_var = P_v1 + P_v2 + P_v3;

  U[index] = (U[index] - tau * div_var + lt * Input[index]) / (1.0f + lt);

  return;
}

extern "C" __global__ void PDnonneg3D_kernel(float *Output, int dimX, int dimY, int dimZ)
{
  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  const long j = blockDim.y * blockIdx.y + threadIdx.y;
  const long k = blockDim.z * blockIdx.z + threadIdx.z;

  if (i >= dimX || j >= dimY || k >= dimZ)
    return;
  long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);

  if (index < static_cast<long long>(dimX * dimY * dimZ))
  {
    if (Output[index] < 0.0f)
      Output[index] = 0.0f;
  }
}
extern "C" __global__ void PDcopy_kernel2D(float *Input, float *Output, int dimX, int dimY)
{
  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  const long j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i >= dimX || j >= dimY)
    return;
  long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j);

  if (index < static_cast<long long>(dimX * dimY))
  {
    Output[index] = Input[index];
  }
}

extern "C" __global__ void PDcopy_kernel3D(float *Input, float *Output, int dimX, int dimY, int dimZ)
{
  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  const long j = blockDim.y * blockIdx.y + threadIdx.y;
  const long k = blockDim.z * blockIdx.z + threadIdx.z;

  if (i >= dimX || j >= dimY || k >= dimZ)
    return;
  long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);

  if (index < static_cast<long long>(dimX * dimY * dimZ))
  {
    Output[index] = Input[index];
  }
}

extern "C" __global__ void getU2D_kernel(float *Input, float *Input_old, float theta, int dimX, int dimY)
{
  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  const long j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i >= dimX || j >= dimY)
    return;
  long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j);

  if (index < static_cast<long long>(dimX * dimY))
  {
    Input[index] += theta * (Input[index] - Input_old[index]);
  }
}

extern "C" __global__ void getU3D_kernel(float *Input, float *Input_old, float theta, int dimX, int dimY, int dimZ)
{
  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  const long j = blockDim.y * blockIdx.y + threadIdx.y;
  const long k = blockDim.z * blockIdx.z + threadIdx.z;

  if (i >= dimX || j >= dimY || k >= dimZ)
    return;
  long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);

  if (index < static_cast<long long>(dimX * dimY * dimZ))
  {
    Input[index] += theta * (Input[index] - Input_old[index]);
  }
}

extern "C" __global__ void PDResidCalc2D_kernel(float *Input1, float *Input2, float *Output, int dimX, int dimY)
{
  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  const long j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i >= dimX || j >= dimY)
    return;
  long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j);

  if (index < static_cast<long long>(dimX * dimY))
  {
    Output[index] = Input1[index] - Input2[index];
  }
}

extern "C" __global__ void PDResidCalc3D_kernel(float *Input1, float *Input2, float *Output, int dimX, int dimY, int dimZ)
{
  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  const long j = blockDim.y * blockIdx.y + threadIdx.y;
  const long k = blockDim.z * blockIdx.z + threadIdx.z;

  if (i >= dimX || j >= dimY || k >= dimZ)
    return;
  long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);

  if (index < static_cast<long long>(dimX * dimY * dimZ))
  {
    Output[index] = Input1[index] - Input2[index];
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

// template<bool nonneg>
// extern "C" __global__ void primal_dual_for_total_variation_2D(float *U, float sigma, int N, int M) {}

// extern "C" __global__ void primal_dual_for_total_variation_2D<true>(float *U, float sigma, int N, int M)
// {
//   // calculate each thread global index
//   const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
//   const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

//   if (xIndex >= N || yIndex >= M)
//   {
//     return;
//   }

//   int index = xIndex + N * yIndex;
//   float2 P1_P2 = dualPD(U, sigma, N, M, xIndex, yIndex, index);
// }

// extern "C" __global__ void primal_dual_for_total_variation_2D<false>(float *U, float sigma, int N, int M)
// {
//   // calculate each thread global index
//   const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
//   const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

//   if (xIndex >= N || yIndex >= M)
//   {
//     return;
//   }

//   int index = xIndex + N * yIndex;
//   float2 P1_P2 = dualPD(U, sigma, N, M, xIndex, yIndex, index);
// }