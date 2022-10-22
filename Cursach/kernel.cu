
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
//#include <iostream>
#include <time.h>
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
#include <math.h>
//#include <thrust>
#include <ctime>

//using namespace thrust;
#define PIC 3.1415926535
using namespace std;
/*__global__ void first_position(float* X, float* Y, float* Z, float* Vx, float* Vy, float* Vz, float* Ax, float* Ay, float* Az, float gran, int j) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	X[id] = (float)rand() / RAND_MAX * gran - gran / 2;
	Y[id] = (float)rand() / RAND_MAX * gran - gran / 2;
	Z[id] = (float)rand() / RAND_MAX * gran - gran / 2;
	Vx[id] = 0;
	Vy[id] = 0;
	Vz[id] = 0;
	Ax[id] = 0;
	Ay[id] = 0;
	Az[id] = 0;

}*/
void AccelerationCPU(float *X, float *Y, float *Z, float *Vx, float *Vy, float *Vz, float *Ax, float *Ay, float *Az, float m, float q, int N, int id, float gran, float rasb, float *Bx, float *By, float *Bz) {
	float ax = 0; // ускорение от силы Кулона
	float ax1 = 0;//ускорение от силы Лоренца
	float ay = 0;
	float ay1 = 0;
	float az = 0;
	float az1 = 0;
	float xx, yy, zz, rr;
	Ax[id] = 0;
	Ay[id] = 0;//ускорения предыдущих итераций
	Az[id] = 0;
	for (int i = 0; i < N; i++) {
		if (i != id) {
			xx = X[i] - X[id];
			yy = Y[i] - Y[id];//растояние между частицами
			zz = Z[i] - Z[id];
			rr = sqrt(xx * xx + yy * yy + zz * zz);
			if (rr > 0.01) {
				rr = q * q / (rr * rr * rr) / m;
				ax += rr * xx;
				ay += rr * yy;//для определения направления вклада силы Кулона
				az += rr * zz;

			}
		}
	}
	float xcoor = (X[id] + gran / 2) / gran;
	float ycoor = (Y[id] + gran / 2) / gran;
	float zcoor = (Z[id] + gran / 2) / gran;
	int zid = round(xcoor * (gran / rasb - 1)) * (gran / rasb) * (gran / rasb) + round(ycoor * (gran / rasb - 1)) * gran / rasb + round(zcoor * (gran / rasb - 1));//индекс магнитного поля
	ax1 = q / m * (Vy[id] * Bz[zid] - By[zid] * Vy[id]);
	ay1 = -q / m * (Vx[id] * Bz[zid] - Vz[id] * Bx[zid]);// раскрытое векторное произведение для силы лоренца F = q[v;B]
	az1 = q / m * (Vx[id] * Bz[zid] - Vz[id] * Bx[zid]);
	Ax[id] = ax + ax1;
	Ay[id] = ay + ay1;//cумма
	Az[id] = az + az1;
}

void PositionCPU(float* X, float* Y, float* Z, float* Vx, float* Vy, float* Vz, float* Ax, float* Ay, float* Az, float tau, int nt, int Np, float gran, int id) {
	X[id + nt * Np] = X[id + (nt - 1) * Np] + Vx[id] * tau + Ax[id] * tau * tau * 0.5;
	Y[id + nt * Np] = Y[id + (nt - 1) * Np] + Vy[id] * tau + Ay[id] * tau * tau * 0.5;  //пересчет новой координаты
	Z[id + nt * Np] = Z[id + (nt - 1) * Np] + Vz[id] * tau + Az[id] * tau * tau * 0.5;
	Vx[id] += Ax[id] * tau;
	Vy[id] += Ay[id] * tau; //вклад скорости от каждой проекции ускорения
	Vz[id] += Az[id] * tau;
	///!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! БАМПИНГ
	if (X[id + (nt - 1) * Np] + Vx[id] * tau + Ax[id] * tau * tau * 0.5 > gran / 2) {
		float obr = X[id + (nt - 1) * Np] + Vx[id] * tau + Ax[id] * tau * tau * 0.5 - gran / 2;
		X[id + nt * Np] = gran / 2 - obr;
		Vx[id] = -1.0 *Vx[id];
	}
	else if (X[id + (nt - 1) * Np] + Vx[id] * tau + Ax[id] * tau * tau * 0.5 < -gran / 2) {
		float obr = -gran / 2 - X[id + (nt - 1) * Np] + Vx[id] * tau + Ax[id] * tau * tau * 0.5;
		X[id + nt * Np] = -gran / 2 + obr;
		
		Vx[id] = -1.0 *Vx[id];
	}
	else if (Y[id + (nt - 1) * Np] + Vy[id] * tau + Ay[id] * tau * tau * 0.5 > gran / 2) {
		float obr = Y[id + (nt - 1) * Np] + Vy[id] * tau + Ay[id] * tau * tau * 0.5 - gran / 2;
		Y[id + nt * Np] = gran / 2 - obr;
		
		Vy[id] = -1.0 * Vy[id];

	}
	else if (Y[id + (nt - 1) * Np] + Vy[id] * tau + Ay[id] * tau * tau * 0.5 < -gran / 2) {
		float obr = -gran / 2 - Y[id + (nt - 1) * Np] + Vy[id] * tau + Ay[id] * tau * tau * 0.5;
		Y[id + nt * Np] = -gran / 2 + obr;
		Vy[id] = -1.0*Vy[id];
	}
	else if (Z[id + (nt - 1) * Np] + Vz[id] * tau + Az[id] * tau * tau * 0.5 > gran / 2) {
		float obr = Z[id + (nt - 1) * Np] + Vz[id] * tau + Az[id] * tau * tau * 0.5 - gran / 2;
		Z[id + nt * Np] = gran / 2 - obr;
		Vz[id] = -1.0*Vz[id];
	}
	else if (Z[id + (nt - 1) * Np] + Vz[id] * tau + Az[id] * tau * tau * 0.5 < -gran / 2) {
		float obr = -gran / 2 - Z[id + (nt - 1) * Np] + Vz[id] * tau + Az[id] * tau * tau * 0.5;
		Z[id + nt * Np] = -gran / 2 + obr;
		Vz[id] = -1.0*Vz[id];
	}


}


__global__ void Acceleration(float* X, float* Y, float* Z, float* Vx, float* Vy, float* Vz, float* Ax, float* Ay, float* Az, float q, int N, float m, float* Bx, float* By, float* Bz, float gran, float rasb, int N_block) {
	int id = threadIdx.x + blockIdx.x * blockDim.x; // разбиение на потоки и блоки, проводящие вычисления
	float ax = 0; // ускорение от силы Кулона
	float ax1 = 0;//ускорение от силы Лоренца
	float ay = 0;
	float ay1 = 0;
	float az = 0;
	float az1 = 0;
	Ax[id] = 0;
	Ay[id] = 0;//ускорения предыдущих итераций
	Az[id] = 0;
	float xxx = X[id];
	float yyy = Y[id];
	float zzz = Z[id];
	__shared__ float Xs[256];
	__shared__ float Ys[256]; 
	__shared__ float Zs[256];
	float xx, yy, zz, rr;
	for (int j = 0; j < N_block; j++) {
		Xs[threadIdx.x] = X[threadIdx.x + j * blockDim.x];
		Ys[threadIdx.x] = Y[threadIdx.x + j * blockDim.x]; 
		Zs[threadIdx.x] = Z[threadIdx.x + j * blockDim.x];

		for (int i = 0; i < blockDim.x; i++) {
			if ((i + j* blockDim.x )!= id) {
				xx = X[i] - xxx;
				yy = Y[i] - yyy;//растояние между частицами
				zz = Z[i] - zzz;
				rr = sqrt(xx * xx + yy * yy + zz * zz);
				if (rr > 0.01) {
					rr = q * q / (rr * rr * rr) / m;
					ax += rr * xx;
					ay += rr * yy;//для определения направления вклада силы Кулона
					az += rr * zz;
				}
			}
		}
		__syncthreads();
	}
	float xcoor = (X[id] + gran / 2) / gran;
	float ycoor = (Y[id] + gran / 2) / gran;
	float zcoor = (Z[id] + gran / 2) / gran;
	int zid = round(xcoor * (gran / rasb - 1)) * (gran / rasb) * (gran / rasb) + round(ycoor * (gran / rasb - 1)) * gran / rasb + round(zcoor * (gran / rasb - 1));//индекс магнитного поля
	ax1 = q / m * (Vy[id] * Bz[zid] - By[zid] * Vy[id]);
	ay1 = -q / m * (Vx[id] * Bz[zid] - Vz[id] * Bx[zid]);// раскрытое векторное произведение для силы лоренца F = q[v;B]
	az1 = q / m * (Vx[id] * Bz[zid] - Vz[id] * Bx[zid]);
	Ax[id] = ax+ax1;
	Ay[id] = ay+ay1;//cумма
	Az[id] = az+az1;
}  
	__global__ void Position(float* X, float* Y, float* Z, float* Vx, float* Vy, float* Vz, float* Ax, float* Ay, float* Az, float tau, int nt, int Np,float gran) {
			int id = threadIdx.x + blockIdx.x * blockDim.x;//выделение потоков и блоков под пересчет координат
			X[id + nt * Np] = X[id + (nt - 1) * Np] + Vx[id] * tau + Ax[id] * tau * tau * 0.5;
			Y[id + nt * Np] = Y[id + (nt - 1) * Np] + Vy[id] * tau + Ay[id] * tau * tau * 0.5;  //пересчет новой координаты
			Z[id + nt * Np] = Z[id + (nt - 1) * Np] + Vz[id] * tau + Az[id] * tau * tau * 0.5;
			Vx[id] += Ax[id] * tau;
			Vy[id] += Ay[id] * tau; //вклад скорости от каждой проекции ускорения
			Vz[id] += Az[id] * tau;
		///!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! БАМПИНГ
			if (X[id + (nt - 1) * Np] + Vx[id] * tau + Ax[id] * tau * tau * 0.5 > gran / 2) {
				float obr = X[id + (nt - 1) * Np] + Vx[id] * tau + Ax[id] * tau * tau * 0.5 - gran / 2;
				X[id + nt * Np] = gran / 2 - obr;
				Vx[id] = -Vx[id];
			}
				else if (X[id + (nt - 1) * Np] + Vx[id] * tau + Ax[id] * tau * tau * 0.5 < -gran / 2) {
					float obr = -gran / 2 - X[id + (nt - 1) * Np] + Vx[id] * tau + Ax[id] * tau * tau * 0.5;
					X[id + nt * Np] = -gran / 2 + obr;
					Vx[id] = -Vx[id];
				}
				else if (Y[id + (nt - 1) * Np] + Vy[id] * tau + Ay[id] * tau * tau * 0.5 > gran / 2) {
					float obr = Y[id + (nt - 1) * Np] + Vy[id] * tau + Ay[id] * tau * tau * 0.5 - gran / 2;
					Y[id + nt * Np] = gran / 2 - obr;
					Vy[id] = -Vy[id];

				}
				else if (Y[id + (nt - 1) * Np] + Vy[id] * tau + Ay[id] * tau * tau * 0.5 < -gran / 2) {
					float obr = -gran / 2 - Y[id + (nt - 1) * Np] + Vy[id] * tau + Ay[id] * tau * tau * 0.5;
					Y[id + nt * Np] = -gran / 2 + obr;
					Vy[id] = -Vy[id];
				}
				else if (Z[id + (nt - 1) * Np] + Vz[id] * tau + Az[id] * tau * tau * 0.5 > gran / 2) {
					float obr = Z[id + (nt - 1) * Np] + Vz[id] * tau + Az[id] * tau * tau * 0.5 - gran / 2;
					Z[id + nt * Np] = gran / 2 - obr;
					Vz[id] = -Vz[id];
				}
				else if (Z[id + (nt - 1) * Np] + Vz[id] * tau + Az[id] * tau * tau * 0.5 < -gran / 2) {
					float obr = -gran / 2 - Z[id + (nt - 1) * Np] + Vz[id] * tau + Az[id] * tau * tau * 0.5;
					Z[id + nt * Np] = -gran / 2 + obr;
					Vz[id] = -Vz[id];
				}


	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	
}






float chislInt1(float x, float y, float z, float radius, float IoM) {//Интеграл симпсона для Bx , By
	float a = 0, b = 2 * PIC, eps = 0.1;//Нижний и верхний пределы интегрирования (a, b), погрешность (eps).
	float I = eps + 1, I1 = 0;//I-предыдущее вычисленное значение интеграла, I1-новое, с большим N.
	y = pow(z * z + y * y, 1 / 2);
	for (float N = 2; (N <= 4) || (fabs(I1 - I) > eps); N *= 2)
	{
		float h, sum2 = 0, sum4 = 0, sum = 0;
		h = (b - a) / (2 * N);//Шаг интегрирования.
		for (float i = 1; i <= 2 * N - 1; i += 2)
		{
			sum4 += (IoM * radius * (z * cos(a+h*i)) / pow((radius * radius + x * x + y * y - 2 * radius * y * cos(a+h*i)), 3 / 2));//Int1funcX(x, y, z, radius, I, a + h * i);//Значения с нечётными индексами, которые нужно умножить на 4.
			sum2 += (IoM * radius * (z * cos(a+h*(i+1))) / pow((radius * radius + x * x + y * y - 2 * radius * y * cos(a+h*(i+1))), 3 / 2));//Int1funcX(x, y, z, radius, I, a + h * (i + 1));//Значения с чётными индексами, которые нужно умножить на 2.
		}
		sum = (IoM * radius * (z * cos(a)) / pow((radius * radius + x * x + y * y - 2 * radius * y * cos(a)), 3 / 2)) + 4 * sum4 + 2 * sum2 - (IoM * radius * (z * cos(b)) / pow((radius * radius + x * x + y * y - 2 * radius * y * cos(b)), 3 / 2));//Отнимаем значение f(b) так как ранее прибавили его дважды. 
		I = I1;
		I1 = (h / 3) * sum;
	}
	return I1;
}//конечно - разностная сетка с центром в в центре витка
float chislInt2(float x, float y, float z, float radius , float IoM) {//Интеграл симпсона для Bx , By
	float a = 0, b = 2 * PIC, eps = 0.1;//Нижний и верхний пределы интегрирования (a, b), погрешность (eps).
	float I = eps + 1, I1 = 0;//I-предыдущее вычисленное значение интеграла, I1-новое, с большим N.
	y = pow(z * z + y * y, 1 / 2);
	for (float N = 2; (N <= 4) || (fabs(I1 - I) > eps); N *= 2)
	{
		float h, sum2 = 0, sum4 = 0, sum = 0;
		h = (b - a) / (2 * N);//Шаг интегрирования.
		for (float i = 1; i <= 2 * N - 1; i += 2)
		{
			sum4 += IoM * radius * (radius - y * cos(a+h*i)) / pow((radius * radius + x * x + y * y - 2 * radius * y * cos(a+h*i)), 3 / 2);//Значения с нечётными индексами, которые нужно умножить на 4.
			sum2 += IoM * radius * (radius - y * cos(a+h*(i+1))) / pow((radius * radius + x * x + y * y - 2 * radius * y * cos(a+h*(i+1))), 3 / 2);//Значения с чётными индексами, которые нужно умножить на 2.
		}
		sum = IoM * radius * (radius - y * cos(a)) / pow((radius * radius + x * x + y * y - 2 * radius * y * cos(a)), 3 / 2) + 4 * sum4 + 2 * sum2 - IoM * radius * (radius - y * cos(b)) / pow((radius * radius + x * x + y * y - 2 * radius * y * cos(b)), 3 / 2);//Отнимаем значение f(b) так как ранее прибавили его дважды. 
		I = I1;
		I1 = (h / 3) * sum;
	}
	return I1;
}
FILE* out;

//device_vector<vector<float>> X;


int main(int argc, char* agrv[]) {
	
	srand(time(NULL));
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int N= 256;
	int NT = 500;
	float m = 1.0;
	float gran = 1000.0;
	float dlina = 100.0;
	float Radius = 10.0;
	float I = 100;
	float q = 90.0;     
	float oblsozd = 1000.0;
	

	float tau = 0.1;
	float *hX, * hY, * hZ, * hVx, * hVy, * hVz, * hAx, * hAy, * hAz;
	unsigned int memory_size = sizeof(float) * N;
	unsigned int memory_size_big = sizeof(float) * NT * N;
	int N_thread = 256;
	int N_blocks = N / N_thread;
	
	hX = (float*) malloc(memory_size_big);
	hY = (float*) malloc (memory_size_big);
	hZ = (float*)malloc (memory_size_big);
	hVx = (float*)malloc(memory_size);
	hVy = (float*)malloc(memory_size);
	hVz = (float*)malloc(memory_size);
	hAx = (float*)malloc(memory_size);
	hAy = (float*)malloc(memory_size);
	hAz = (float*)malloc(memory_size);


	for (int j = 0; j < N; j++) {
		//first_position <<<N_blocks, N_thread>>> (hX, hY, hZ, hVx, hVy, hVz, hAx, hAy, hAz, gran, j);
		


		hX[j] = (float)rand() / RAND_MAX*oblsozd-oblsozd/2;
		//printf("%f \n", hX[j]);
		//phi = (float)rand() / (RAND_MAX) * 10000000;
		//phi = cos(phi);
		hY[j] = (float)rand() / RAND_MAX  * oblsozd - oblsozd / 2;
		hZ[j] = (float)rand() / RAND_MAX * oblsozd - oblsozd / 2;
		
		hVx[j] = 0;
		hVy[j] = 0;
		hVz[j] = 0;
		hAx[j] = 0;
		hAy[j] = 0;
		hAz[j] = 0;
	}
	float* dX, * dY, * dZ, *dVx, *dVy, *dVz, *dAx, *dAy, *dAz;
	cudaMalloc((void**)&dX, memory_size_big);
	cudaMalloc((void**)&dY, memory_size_big);
	cudaMalloc((void**)&dZ, memory_size_big);
	cudaMalloc((void**)&dVx, memory_size);
	cudaMalloc((void**)&dVy, memory_size);
	cudaMalloc((void**)&dVz, memory_size);
	cudaMalloc((void**)&dAx, memory_size);
	cudaMalloc((void**)&dAy, memory_size);
	cudaMalloc((void**)&dAz, memory_size);
	cudaMemcpy(dX, hX, memory_size_big, cudaMemcpyHostToDevice);
	cudaMemcpy(dY, hY, memory_size_big, cudaMemcpyHostToDevice);
	cudaMemcpy(dZ, hZ, memory_size_big, cudaMemcpyHostToDevice);
	cudaMemcpy(dVx, hVx, memory_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dVy, hVy, memory_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dVz, hVz, memory_size, cudaMemcpyHostToDevice);
	

	
	unsigned int memsize = sizeof(float) * gran * gran * gran / dlina / dlina / dlina;
	//printf("%i", memsize);
	float* hBx, * hBy, * hBz;
	float *dBx,*dBy,*dBz;
	hBx = (float*)malloc(memsize);
	hBy = (float*)malloc(memsize);
	hBz = (float*)malloc(memsize);

	cudaMalloc((void**) &dBx, memsize);
	cudaMalloc((void**) &dBy, memsize);
	cudaMalloc((void**) &dBz, memsize);
	
	//cudaMemcpy(dBx, hBx, memsize, cudaMemcpyHostToDevice);
	//cudaMemcpy(dBy, hBy, memsize, cudaMemcpyHostToDevice);
	//cudaMemcpy(dBz, hBz, memsize, cudaMemcpyHostToDevice);
	int z = gran / dlina;
	
	for (int i = 0; i < z; i++) {
		for (int j = 0; j < z; j++) {
			for (int k = 0; k < z; k++) {
				//float buf1 = (chislInt1(i * dlina - gran / 2 + dlina / 2, j * dlina - gran / 2 + dlina / 2, k * dlina - gran / 2 + dlina / 2, Radius, I));
				//printf("%d", buf1);
				//float buf2 = (chislInt2(i * dlina - gran / 2 + dlina / 2, j * dlina - gran / 2 + dlina / 2, k * dlina - gran / 2 + dlina / 2, Radius, I));
				//printf("%d", buf2);
					//dBx[0] = 1;
				//hBx[1] = 1;
				hBx[z*z*i+z*j+k] =(chislInt1(i*dlina-gran/2+dlina/2, j * dlina - gran / 2 + dlina / 2, k * dlina - gran / 2 + dlina / 2, Radius, I));
				hBy[z*z*i+z*j+k] =(chislInt2(i * dlina - gran / 2 + dlina / 2, j * dlina - gran / 2 + dlina / 2, k * dlina - gran / 2 + dlina / 2, Radius, I));
				hBz[z * z * i + z * j + k] = 0;
			}
		}
			
	}


	cudaMemcpy(dBx, hBx, memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(dBy, hBy, memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(dBz, hBz, memsize, cudaMemcpyHostToDevice);
	//printf("%d", &dBx[0]);
	//cudaThreadSynchronize();
	
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start);
	unsigned int start_time = clock();
	for (int j = 1; j < NT; j++) {
		printf("CPU \n %i",j);
		for (int id = 0; id < N; id++) {
			AccelerationCPU(hX, hY, hZ, hVx, hVy, hVz, hAx, hAy, hAz, m, q, N, id, gran, dlina, hBx, hBy, hBz);
		}
		for (int id = 0; id < N; id++) {
			PositionCPU(hX, hY, hZ, hVx, hVy, hVz, hAx, hAy, hAz, tau, j, N, gran, id);
		}
	
	
	}
	unsigned int end_time = clock();
	unsigned int search_time = end_time - start_time;
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	float milliseconds1 = 0;
	//cudaEventElapsedTime(&milliseconds1, start, stop);
	printf("CPUtime %f", milliseconds1);
	
	
	cudaEventRecord(start);

	for (int j = 1; j < NT; j++) {
		printf("\n %i ",j);
		//Bumping << <N_blocks, N_thread >> > (dX, dY, dZ, dVx, dVy, dVz, gran);
		//cudaThreadSynchronize();
		Acceleration <<< N_blocks, N_thread >>> (dX, dY, dZ,dVx, dVy, dVz, dAx, dAy, dAz, q, N, m, dBx, dBy, dBz, gran, dlina, N_blocks);
		cudaThreadSynchronize();
		Position <<<N_blocks, N_thread >> > (dX, dY, dZ, dVx, dVy, dVz, dAx, dAy, dAz, tau, j, N, gran);
		cudaThreadSynchronize();
		
	}

	cudaMemcpy(hX, dX, memory_size_big, cudaMemcpyDeviceToHost);
	cudaMemcpy(hY, dY, memory_size_big, cudaMemcpyDeviceToHost);
	cudaMemcpy(hZ, dZ, memory_size_big, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
//	printf("%e", hX[10000]);
	

	//printf("%e", hX[10000]);
	cudaEventRecord(stop);
	//float timerValueGPU = 0;
	
	
	
	out = fopen("coors.txt", "w");
	for (int nt = 0; nt < NT; nt++) {
		for (int j = 0; j < N; j++) {
			fprintf(out,"%e, %e, %e\n", hX[j+nt*N], hY[j+nt*N], hZ[j+nt*N]);
		}
		
	
	}
	fcloseall();

	printf("CPUTIME = %i \n", search_time);
	printf("GPUTIME = %f \n", milliseconds);

	//cudaEventDestroy(start);
	//cudaEventDestroy(end);
	free(hX);
	free(hY);
	free(hZ);
	free(hVx);
	free(hVy);
	free(hVz);
	free(hAx);
	free(hAy);
	free(hAz);
	cudaFree(dX);
	cudaFree(dY);
	cudaFree(dZ);
	cudaFree(dVx);
	cudaFree(dVy);
	cudaFree(dVz);
	cudaFree(dAx);
	cudaFree(dAy);
	cudaFree(dAz);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	



}