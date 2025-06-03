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

// Original 2D
extern "C" __global__ void dualPD_kernel(float *U, float *P1, float *P2, float sigma, int N, int M)
{

  // calculate each thread global index
  const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

  int index = xIndex + N * yIndex;

  if ((xIndex < N) && (yIndex < M))
  {
    if (xIndex == N - 1)
      P1[index] += sigma * (U[(xIndex - 1) + N * yIndex] - U[index]);
    else
      P1[index] += sigma * (U[(xIndex + 1) + N * yIndex] - U[index]);
    if (yIndex == M - 1)
      P2[index] += sigma * (U[xIndex + N * (yIndex - 1)] - U[index]);
    else
      P2[index] += sigma * (U[xIndex + N * (yIndex + 1)] - U[index]);
  }
  return;
}
extern "C" __global__ void Proj_funcPD2D_iso_kernel(float *P1, float *P2, int N, int M)
{

  float denom;
  // calculate each thread global index
  const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

  int index = xIndex + N * yIndex;

  if ((xIndex < N) && (yIndex < M))
  {
    denom = P1[index] * P1[index] + P2[index] * P2[index];
    if (denom > 1.0f)
    {
      P1[index] = P1[index] / sqrtf(denom);
      P2[index] = P2[index] / sqrtf(denom);
    }
  }
  return;
}
extern "C" __global__ void Proj_funcPD2D_aniso_kernel(float *P1, float *P2, int N, int M)
{

  float val1, val2;
  // calculate each thread global index
  const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

  int index = xIndex + N * yIndex;

  if ((xIndex < N) && (yIndex < M))
  {
    val1 = abs(P1[index]);
    val2 = abs(P2[index]);
    if (val1 < 1.0f)
    {
      val1 = 1.0f;
    }
    if (val2 < 1.0f)
    {
      val2 = 1.0f;
    }
    P1[index] = P1[index] / val1;
    P2[index] = P2[index] / val2;
  }
  return;
}
extern "C" __global__ void DivProj2D_kernel(float *U, float *Input, float *P1, float *P2, float lt, float tau, int N, int M)
{
  float P_v1, P_v2, div_var;

  // calculate each thread global index
  const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

  int index = xIndex + N * yIndex;

  if ((xIndex < N) && (yIndex < M))
  {
    if (xIndex == 0)
      P_v1 = -P1[index];
    else
      P_v1 = -(P1[index] - P1[(xIndex - 1) + N * yIndex]);
    if (yIndex == 0)
      P_v2 = -P2[index];
    else
      P_v2 = -(P2[index] - P2[xIndex + N * (yIndex - 1)]);
    div_var = P_v1 + P_v2;
    U[index] = (U[index] - tau * div_var + lt * Input[index]) / (1.0 + lt);
  }
  return;
}
extern "C" __global__ void PDnonneg2D_kernel(float *Output, int dimX, int dimY)
{
  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  const long j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i >= dimX || j >= dimY)
    return;
  long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j);

  if (index < static_cast<long long>(dimX * dimY))
  {
    if (Output[index] < 0.0f)
      Output[index] = 0.0f;
  }
}
/************************************************/
/*****************3D modules*********************/
/************************************************/
__device__ void dualPD3D(float *U, float *P1, float *P2, float *P3, float sigma, int dimX, int dimY, int dimZ, long i, long j, long k, long long index)
{
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
}
__device__ void Proj_funcPD3D_iso(float *P1, float *P2, float *P3, long long index)
{
  float denom = P1[index] * P1[index] + P2[index] * P2[index] + P3[index] * P3[index];
  if (denom > 1.0f)
  {
    float sq_denom = 1.0f / sqrtf(denom);
    P1[index] *= sq_denom;
    P2[index] *= sq_denom;
    P3[index] *= sq_denom;
  }
}
__device__ void Proj_funcPD3D_aniso(float *P1, float *P2, float *P3, long long index)
{
  float val1 = abs(P1[index]);
  float val2 = abs(P2[index]);
  float val3 = abs(P3[index]);

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
}

__device__ void DivProj3D(float *U, float *Input, float *P1, float *P2, float *P3, float tau, float lt, int dimX, int dimY, long i, long j, long k, long long index)
{
  float P_v1, P_v2, P_v3;

  if (i == 0)
  {
    P_v1 = -P1[index];
  }
  else
  {
    long long index1 = static_cast<long long>(i - 1) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);
    P_v1 = -(P1[index] - P1[index1]);
  }

  if (j == 0)
  {
    P_v2 = -P2[index];
  }
  else
  {
    long long index2 = static_cast<long long>(i) + dimX * static_cast<long long>(j - 1) + dimX * dimY * static_cast<long long>(k);
    P_v2 = -(P2[index] - P2[index2]);
  }

  if (k == 0)
  {
    P_v3 = -P3[index];
  }
  else
  {
    long long index3 = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k - 1);
    P_v3 = -(P3[index] - P3[index3]);
  }

  float div_var = P_v1 + P_v2 + P_v3;

  U[index] = (U[index] - tau * div_var + lt * Input[index]) / (1.0f + lt);
}

extern "C" __global__ void primal_dual_for_total_variation_3D(float *U, float *old_U, float* Input, float *P1, float *P2, float *P3, float sigma, float tau, float lt, float theta, int dimX, int dimY, int dimZ)
{
  // calculate each thread global index
  const long xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const long yIndex = blockIdx.y * blockDim.y + threadIdx.y;
  const long zIndex = blockIdx.z * blockDim.z + threadIdx.z;

  if (xIndex >= dimX || yIndex >= dimY || zIndex >= dimZ)
  {
    return;
  }

  long long index = static_cast<long long>(xIndex) + dimX * static_cast<long long>(yIndex) + dimX * dimY * static_cast<long long>(zIndex);

  // dualPD3D(U, P1, P2, P3, sigma, dimX, dimY, dimZ, xIndex, yIndex, zIndex, index);

  // if (nonneg && U[index] < 0.0f)
  // {
  //   U[index] = 0.0f;
  // }

  // if (run_isotropic_kernel)
  // {
  //   Proj_funcPD3D_iso(P1, P2, P3, index);
  // }
  // else
  // {
  //   Proj_funcPD3D_aniso(P1, P2, P3, index);
  // }

  // float old_U = U[index];

  DivProj3D(U, Input, P1, P2, P3, tau, lt, dimX, dimY, xIndex, yIndex, zIndex, index);

  U[index] += theta * (U[index] - old_U[index]);
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