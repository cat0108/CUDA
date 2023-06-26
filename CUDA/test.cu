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
		//���Ƚ�ȫ��Ԫ����Ϊ0���Խ���Ԫ����Ϊ1
		for (int j = 0; j < N; j++)
		{
			data[i * N + j] = 0;

		}
		data[i * N + i] = 1.0;
		//�������ǵ�λ�ó�ʼ��Ϊ�����
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

//��������,ÿ���̼߳���һ�γ���
__global__
void division_Kernel(float* data, int k, int N) {
	//�߳�����ΪblockDim*blockIdx+threadidx�����������̣߳�ÿ���߳̽���һ�β���
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	while (k + tid < N)
	{
		data[k * N + tid] = data[k * N + tid] / data[k * N + k];
		tid += blockDim.x * gridDim.x;//����߳��������ã������һ�Σ����߳�����
	}
}
__global__
void eliminate_Kernel(float* data, int k, int N) {
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	if (tx == 0)
		data[k * N + k] = 1.0;
	int row = k + 1 + blockIdx.x;//ÿ���鸺��һ��,����ÿ���������
	while (row < N) {
		int tid = threadIdx.x;//���ڵ�ÿһ�����̸���һ����ȥ����
		while (k + 1 + tid < N) {
			int col = k + 1 + tid;
			data[row * N + col] = data[row * N + col] - data[row * N + k] * data[k * N + col];
			tid += blockDim.x;//���û�����꣬������
		}
		__syncthreads();//�����߳�ͬ��
		if (threadIdx.x == 0)
			data[row * N + k] = 0;
		row += gridDim.x;//�߳̿鲻����������һ��
	}
}

void cudaEliminate(float* data, int N)
{
	//������
	cudaError_t ret;
	int grid = 128;
	int block = 128;
	float* gpudata;
	int size = N * N * sizeof(float);
	//Ϊgpu�˷����ڴ�
	ret = cudaMalloc(&gpudata, size);
	if (ret != cudaSuccess)
		cout << "cudaMalloc data failed" << endl;
	//���ݿ���
	ret = cudaMemcpy(gpudata, data, size, cudaMemcpyHostToDevice);
	if (ret != cudaSuccess)
		cout << "cudaMemcpyHostToDevice failed" << endl;
	dim3 dimBlock(block, 1);//�߳̿�
	dim3 dimGrid(grid, 1);//�߳�����
	cudaEvent_t start, stop;//��ʱ
	float elapseTime = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);//��ʼ��ʱ


	for (int k = 0; k < N; k++)
	{
		division_Kernel << <dimGrid, dimBlock >> > (gpudata, k, N);
		cudaDeviceSynchronize();//ͬ��
		eliminate_Kernel << <dimGrid, dimBlock >> > (gpudata, k, N);
		cudaDeviceSynchronize();
	}
	//���ݴ����cpu��		
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