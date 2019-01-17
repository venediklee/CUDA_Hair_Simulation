#pragma once
#include "cuda.h"
#include "cuda_device_runtime_api.h"
#include "cuda_runtime.h"

#include "HairSimCPU.h"

//Macro for checking cuda errors following a cuda launch or api call--wrapper function
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
//kernel<<<1,1>>>(a_d);
//gpuErrchk(cudaPeekAtLastError());
//gpuErrchk(cudaMemcpy(a_h, a_d, size * sizeof(int), cudaMemcpyDeviceToHost));



void SaveOriginalSettingsGPU(float headRadius, int hairCount, float hairLength);

void HairPointASetterGPU(int hairCount, float headRadius, hair * hairPoints);

void ApplyWindV0(hair *hairPoints, wind w, int blockSize, int blockCount, int smoothing);

__global__ void StartWindZV2(hair * hairPoints, char sign, float strength);

__global__ void StartWindZV1(hair * hairPoints, char sign, float strength);

__global__ void StartWindZV0(hair *hairPoints, char sign, float strength);

__global__ void CollisionDetectionV1(hair * hairPoints);

__global__ void CollisionDetectionV0(hair* hairPoints);

void FreeAllGPU(hair *hairPoints);
