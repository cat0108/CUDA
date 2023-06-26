#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include <iomanip>
#include<cuda.h>
#include<Windows.h>
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
using namespace std;
void Initialize(int N, float* data)
{
	for (int i = 0; i < N; i++)
	{
		//首先将全部元素置为0，对角线元素置为1
		for (int j = 0; j < N; j++)
		{
			data[i * N + j] = 0;

		}
		data[i * N + i] = 1.0;
		//将上三角的位置初始化为随机数
		for (int j = i + 1; j < N; j++)
		{
			data[i * N + j] = rand();
		}
	}
	for (int k = 0; k < N; k++)
	{
		for (int i = k + 1; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				data[i * N + j] += data[k * N + j];

			}
		}
	}
}

//除法操作,每个线程计算一次除法
__global__
void division_Kernel(float* data, int k, int N) {
	//线程索引为blockDim*blockIdx+threadidx，调用所有线程，每个线程进行一次操作
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	while (k + tid < N)
	{
		data[k * N + tid] = data[k * N + tid] / data[k * N + k];
		tid += blockDim.x * gridDim.x;//如果线程数不够用，多计算一次，加线程总数
	}
}
__global__
void eliminate_Kernel(float* data, int k, int N) {
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	if (tx == 0)
		data[k * N + k] = 1.0;
	int row = k + 1 + blockIdx.x;//每个块负责一行,这是每个块的索引
	while (row < N) {
		int tid = threadIdx.x;//块内的每一个进程负责一次消去操作
		while (k + 1 + tid < N) {
			int col = k + 1 + tid;
			data[row * N + col] = data[row * N + col] - data[row * N + k] * data[k * N + col];
			tid += blockDim.x;//如果没有做完，接着做
		}
		__syncthreads();//块内线程同步
		if (threadIdx.x == 0)
			data[row * N + k] = 0;
		row += gridDim.x;//线程块不够，进行下一轮
	}
}

void cudaEliminate(float* data, int N)
{
	//错误检查
	cudaError_t ret;
	int grid = 128;
	int block = 128;
	float* gpudata;
	int size = N * N * sizeof(float);
	//为gpu端分配内存
	ret = cudaMalloc(&gpudata, size);
	if (ret != cudaSuccess)
		cout << "cudaMalloc data failed" << endl;
	//内容拷贝
	ret = cudaMemcpy(gpudata, data, size, cudaMemcpyHostToDevice);
	if (ret != cudaSuccess)
		cout << "cudaMemcpyHostToDevice failed" << endl;
	dim3 dimBlock(block, 1);//线程块
	dim3 dimGrid(grid, 1);//线程网格
	cudaEvent_t start, stop;//计时
	float elapseTime = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);//开始计时


	for (int k = 0; k < N; k++)
	{
		division_Kernel << <dimGrid, dimBlock >> > (gpudata, k, N);
		cudaDeviceSynchronize();//同步
		eliminate_Kernel << <dimGrid, dimBlock >> > (gpudata, k, N);
		cudaDeviceSynchronize();
	}
	//数据传输回cpu端		
	cudaError_t cudaStatus2 = cudaGetLastError();
	if(cudaStatus2!=cudaSuccess)
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus2));
	ret = cudaMemcpy(data, gpudata, size, cudaMemcpyDeviceToHost);
	if(ret!=cudaSuccess)
		printf("cudaMemcpyDeviceToHost failed!\n");

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapseTime, start, stop);
	cout << "cuda_time= " << elapseTime << " ms" << endl;

	cudaFree(gpudata);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
int main() {
	int N;
	cin >> N;
	float* gdata=new float[10000*10000];
	Initialize(N, gdata);
	cudaEliminate(gdata, N);
	delete gdata;
}