
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <iostream>
#include <bitset>
#include <sstream>
#include <stdlib.h>
#include <vector>
#include <iomanip>

typedef unsigned char BYTE;

using namespace std;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//    int i = threadIdx.x;
//    c[i] = a[i] + b[i];
//}
//
//int main()
//{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}

////////////////////////////////////////////////////
//int S1[4][16] = { { 14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7 },
//{ 0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8 },
//{ 4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0 },
//{ 15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13 } };
//
//int S2[4][16] = { { 15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10 },
//{ 3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5 },
//{ 0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15, },
//{ 13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9 } };
//
//int S3[4][16] = { { 10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8 },
//{ 13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1 },
//{ 13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7 },
//{ 1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12 } };
//
//int S4[4][16] = { { 7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15 },
//{ 13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9 },
//{ 10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4 },
//{ 3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14 } };
//
//int S5[4][16] = { { 2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9 },
//{ 14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6 },
//{ 4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14 },
//{ 11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3 } };
//
//int S6[4][16] = { { 12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11 },
//{ 10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8 },
//{ 9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6 },
//{ 4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13 } };
//
//int S7[4][16] = { { 4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1 },
//{ 13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6 },
//{ 1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2 },
//{ 6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12 } };
//
//int S8[4][16] = { { 13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7 },
//{ 1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2 },
//{ 7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8 },
//{ 2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11 } };


int PC_1[56] = { 56, 48, 40, 32, 24, 16, 8, 0,
57, 49, 41, 33, 25, 17, 9, 1,
58, 50, 42, 34, 26, 18, 10, 2,
59, 51, 43, 35, 62, 54, 46, 38,
30, 22, 14, 6, 61, 53, 45, 37,
29, 21, 13, 5, 60, 52, 44, 36,
28, 20, 12, 4, 27, 19, 11, 3};


int shifts[] = { 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1 };

int PC_2[] = { 13, 16, 10, 23, 0, 4,
2, 27, 14, 5, 20, 9,
22, 18, 11, 3, 25, 7,
15, 6, 26, 19, 12, 1,
40, 51, 30, 36, 46, 54,
29, 39, 50, 44, 32, 47,
43, 48, 38, 55, 33, 52,
45, 41, 49, 35, 28, 31 };


int IP[] = { 57, 49, 41, 33, 25, 17, 9, 1,
59, 51, 43, 35, 27, 19, 11, 3,
61, 53, 45, 37, 29, 21, 13, 5,
63, 55, 47, 39, 31, 23, 15, 7,
56, 48, 40, 32, 24, 16,  8, 0,
58, 50, 42, 34, 26, 18, 10, 2,
60, 52, 44, 36, 28, 20, 12, 4,
62, 54, 46, 38, 30, 22, 14, 6 };

int E[] = { 31, 0, 1, 2, 3, 4,
3, 4, 5, 6, 7, 8,
7, 8, 9, 10, 11, 12,
11, 12, 13, 14, 15, 16,
15, 16, 17, 18, 19, 20,
19, 20, 21, 22, 23, 24,
23, 24, 25, 26, 27, 28,
27, 28, 29, 30, 31, 0 };
//sprawdziæ s box
int S[8][4][16] = { { {14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7},
{0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8},
{4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0 },
{15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13 } }, 
	{ {15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10},
{3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5 },
{0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15 },
{13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9 } }, 
	{ {10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8},
{13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1 },
{13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7 },
{1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12 } }, 
	{ {7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15},
{13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9},
{10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4},
{3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14} }, 
	{ {2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9},
{14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6},
{4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14},
{11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3} }, 
	{ {12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11},
{10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8 },
{9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6 },
{4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13 } },
	{ {4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1},
{13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6 },
{1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2 },
{6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12 } },
	{ {13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7},
{1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2 },
{7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8 },
{2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11 } }
};

int P[] ={15, 6, 19, 20,
	28, 11, 27, 16,
	0, 14, 22, 25,
	4, 17, 30, 9,
	1, 7, 23, 13,
	31, 26, 2, 8,
	18, 12, 29, 5,
	21, 10, 3, 24};

int IP_1[] = {
39, 7, 47, 15, 55, 23, 63, 31,
38, 6, 46, 14, 54, 22, 62, 30,
37, 5, 45, 13, 53, 21, 61, 29,
36, 4, 44, 12, 52, 20, 60, 28,
35, 3, 43, 11, 51, 19, 59, 27,
34, 2, 42, 10, 50, 18, 58, 26,
33, 1, 41, 9, 49, 17, 57, 25,
32, 0, 40, 8, 48, 16, 56, 24 };

void fun()
{
	for(int i = 0; i < 56; i++)
	{
		cout << PC_1[i] - 1 << ", ";

		if (!(i % 8))
			cout << "\n";
	}
}

template< typename T, size_t N, size_t M >
void printArray(T(&theArray)[N][M], int char_endl_nbr) {
	for (int x = 0; x < N; x++) 
	{
		for (int y = 0; y < M; y++)
		{
			cout << theArray[x][y];
			if (y == char_endl_nbr)
				cout << endl;
		}
		cout << endl;
	}
}


//template< typename T, size_t N, size_t M >
//void printArray2(T(&theArray)[N][M], int char_endl_nbr) {
//	for (int x = 0; x < N; x++)
//	{
//		for (int y = 0; y < M; y++)
//		{
//			cout << theArray[x][y];
//			if (y == char_endl_nbr)
//				cout << endl;
//			if (!(y % 6))
//				cout << " ";
//		}
//		cout << endl;
//	}
//}
template< typename T, size_t N, size_t M >
void printArray2(T(&theArray)[N][M], int char_endl_nbr) {
	for (int x = 0; x < N; x++)
	{
		for (int y = 0; y < M; y++)
		{
			cout << theArray[x][y];
			if (y == char_endl_nbr)
				cout << endl;
//			if (!(y % 6))
//				cout << " ";
		}
		cout << endl;
	}
}

//bitset bytesToBitset<int numBytes>(byte *bytes)
//{
//	std::bitset<numBytes * CHAR_BIT> b;
//
//	for (int i = 0; i < numBytes; ++i)
//	{
//		byte cur = bytes[i];
//		int offset = i * CHAR_BIT;
//
//		for (int bit = 0; bit < CHAR_BIT; ++bit)
//		{
//			b[offset] = cur & 1;
//			++offset;   // Move to next bit in b
//			cur >>= 1;  // Move to next bit in array
//		}
//	}
//
//	return b;
//}

//
//template<int numBytes>
//void bytesToBitset(string key_binary_ret)
//{
////	unsigned char c = 'a';
//
//	char const *c_key = key_binary_ret.c_str();
//
//	for (int i = 0; i < key_binary_ret.size(); i++)
//	{
//		for (int j = 0; j < 8; j++)
//		{
//
//			std::cout << ((c_key[i] >> j) & 1);
//		}
//		cout << " ";
//	}
//	
//}
//
//void bytesToBitset(string key_binary_ret)
//{
//	//	unsigned char c = 'a';
//
//	char const *c_key = key_binary_ret.c_str();
//
//	for (int i = 0; i < key_binary_ret.size(); i++)
//	{
//		for (int j = 0; j < 8; j++)
//		{
//
//			std::cout << ((c_key[i] >> j) & 1);
//		}
//		cout << " ";
//	}
//
//}
//
//int *get_bits(int n, int bitswanted) {
//	int *bits = (int *)malloc(sizeof(int) * bitswanted);
//
//	int k;
//	for (k = 0; k<bitswanted; k++) {
//		int mask = 1 << k;
//		int masked_n = n & mask;
//		int thebit = masked_n >> k;
//		bits[k] = thebit;
//	}
//
//	return bits;
//}
//

//template<int numBytes>
//bitset<numBytes * CHAR_BIT>bytesToBitset(char const *bytes)
//{
////	char const *bytes = key_binary_ret.c_str();
//	bitset<numBytes * CHAR_BIT> b = *bytes;
//
//	for (int i = 1; i < numBytes; ++i)
//	{
//		b <<= CHAR_BIT;  // Move to next bit in array
//		b |= bytes[i];    // Set the lowest CHAR_BIT bits
//	}
//
//	return b;
//}
//
//

void permutePC(int key_binary[], int key_binary_ret[], int key_binary_size, int PC[])
{
	for (int i = 0; i < key_binary_size; i++)
		key_binary_ret[i] = key_binary[PC[i]];

}

//C and D should have 28 array memebers
void createSubkeys(int key[], const int key_size, int C[], int D[], int CD_size, int run_number)
{
	const int size = key_size / 2;
	int tmp_C[28], tmp_D[28];
	for(int i = 0; i < key_size / 2; i++)
	{
		tmp_C[i] = key[i];
		tmp_D[i] = key[i + CD_size];
	}

	for(int i = 0; i < CD_size; i++)
	{
		C[i] = tmp_C[(i + shifts[run_number]) % CD_size];
		D[i] = tmp_D[(i + shifts[run_number]) % CD_size];
	}

}

void decimal2Binary(int decimal_int, int binary_int[], int run_number)
{
	int remainder;

	if (decimal_int <= 1) {
		binary_int[run_number] = decimal_int;
		return;
	}

	remainder = decimal_int % 2;
	decimal2Binary(decimal_int >> 1, binary_int, run_number + 1);
	binary_int[run_number] = remainder;
}

void reverseTab(int tab[], int tab_length)
{
	for (int i = 0; i < tab_length / 2; i++)
	{
		int tmp = tab[i];
		tab[i] = tab[tab_length - i - 1];
		tab[tab_length - i - 1] = tmp;

	}
}

void appendKeys(int leftKey[], int rightKey[], int key_size, int key_ret[])
{
	for(int i = 0; i < key_size; i++)
	{
		key_ret[i] = leftKey[i];
		key_ret[i + key_size] = rightKey[i];
	}
}


void expand(int R[], int tab_ret[], int E[], int E_size)
{

	for (int i = 0; i < E_size; i++)
		tab_ret[i] = R[E[i]];
}

void xor(int first_tab[], int second_tab[], int tab_size, int tab_ret[])
{
	for (int i = 0; i < tab_size; i++)
		tab_ret[i] = (int)(!first_tab[i] != !second_tab[i]);

}

//-->
long long binary2Decimal(int binary_int[], int tab_length)
{
	string int_string = "";

	for (int i = 0; i < tab_length; i++)
		int_string += to_string(binary_int[i]);
	stringstream ss;
	ss << int_string;
	string str = ss.str();
	unsigned long long value = std::stoull(str, 0, 2);
	//std::cout << value << std::endl;
	return value;
}

void f(int R[], int K[], int ret_tab[])
{
	int R_expanded[48];
	expand(R, R_expanded, E, 48);
	//DEBUG
//	for (int i = 0; i < 48; i++)
//	{
//		if (!(i % 6))
//			cout << " ";
//		cout << R_expanded[i];
//	}
//	cout << endl << endl << endl;
	
	int xored[48];
	xor (K, R_expanded, 48, xored);
	//DEBUG
//	for(int i = 0; i < 48; i++)
//	{
//		if (!(i % 6))
//			cout << " ";
//		cout << xored[i];
//	}
//	cout << endl << endl << endl;


	for(int i = 0; i < 8; i++)
	{
		int row[4] = {0, 0, 0, 0 }, column[4] = {0, 0, 0, 0};
		row[3] = xored[6 * i + 5];
		row[2] = xored[6 * i];
		column[0] = xored[6 * i + 1];
		column[1] = xored[6 * i + 2];
		column[2] = xored[6 * i + 3];
		column[3] = xored[6 * i + 4];

		int chunk_length = 4;
		int R_chunk[4] = {0, 0, 0, 0};
		decimal2Binary(S[i][binary2Decimal(row, 4)][binary2Decimal(column, 4)], R_chunk, 0);
		reverseTab(R_chunk, chunk_length);
		for(int j = 0; j < chunk_length; j++)
		{
			R[4 * i + j] = R_chunk[j];
		}

	}
	//DEBUG
//	for(int i = 0; i < 32; i++)
//	{
//		if (!(i % 4))
//			cout << " ";
//		cout << R[i];
//	}
//	cout << endl << endl << endl;
		
	permutePC(R, ret_tab, 32, P);
	//DEBUG
//	for (int i = 0; i < 32; i++)
//	{
//		if (!(i % 4))
//			cout << " ";
//		cout << ret_tab[i];
//	}
//	cout << endl << endl << endl;

}





void reverse(int L[], int R[], int tab_length, int ret_tab[])
{
	for (int i = 0; i < tab_length; i++)
	{
		ret_tab[i] = R[i];
		ret_tab[i + tab_length] = L[i];
	}
}

void messageEncode(int message_binary[], int message_size, int K[][48], int msg_ret[])
{
	int L[32], R[32];
	for(int i = 0; i < message_size / 2; i++)
	{
		L[i] = message_binary[i];
		R[i] = message_binary[i + message_size / 2];
	}

	int prev_L[32], prev_R[32];
	for (int i = 0; i < message_size / 2; i++)
	{
		prev_L[i] = L[i];
		prev_R[i] = R[i];
	}



	for(int i = 0; i < 16; i++)
	{	
		for (int j = 0; j < message_size / 2; j++)
			L[j] = prev_R[j];

		int tmp_f[32];
		f(prev_R, K[i], tmp_f);
		//DEBUG
//		for (int i = 0; i < 32; i++)
//		{
//			if (!(i % 4))
//				cout << " ";
//			cout << tmp_f[i];
//		}
//		cout << endl << endl << endl;

		xor(prev_L, tmp_f, 32, R);

		//DEBUG
//		for(int i = 0; i < 32; i++)
//		{
//			if (!(i % 4))
//				cout << " ";
//			cout << R[i];
//		}
//		cout << endl << endl << endl;

		//przepisanie R i L do prev_R i prev_L
		for(int i = 0; i < message_size / 2; i++)
		{
			prev_L[i] = L[i];
			prev_R[i] = R[i];
		}
	}



	int msg[64];
	reverse(L, R, 32, msg);
	//DEBUG
//	for (int i = 0; i < 64; i++)
//	{
//		if (!(i % 8))
//			cout << " ";
//		cout << msg[i];
//	}

	
	permutePC(msg, msg_ret, 64, IP_1);
	//DEBUG
//	for(int i = 0; i < 64; i++)
//	{
//		if (!(i % 8))
//			cout << " ";
//		cout << msg_ret[i];
//	}	


}	 





//key_binary_ret should be 64 bit long
void desEncyption(int message_binary[], int message_size,int key_binary[], int key_size, int msg_ret[])
{
	int des_block_size_bytes = 8;
	int des_block_size_bits = 64;

//	cout << "omg";
	//DEBUG
//		cout << message.size();
//		cout << "\n" << message << "\n";
//if (message.size() * CHAR_BIT != des_block_size_bits)
//		cout << message.size() * CHAR_BIT;
		

	if (message_size % des_block_size_bytes)
	{
		//int tmp_message_binary[message_size + des_block_size_bytes - (message_size % des_block_size_bytes)]
			cout << "KICIA";
	//	message_binary.append(des_block_size_bytes - (message.size() % des_block_size_bytes), '0');//mayby another char to append  
	}

	//OLD Verwsion with message as string 
//	if (message.size() % des_block_size_bytes)
//		message.append(des_block_size_bytes - (message.size() % des_block_size_bytes), '0');//mayby another char to append  

	//DEBUG
//		cout << "\n" << message << "\n";
//		cout << message.size();

	int key_binary_ret[56];
	permutePC(key_binary, key_binary_ret, sizeof(key_binary_ret) / sizeof(key_binary_ret[0]), PC_1);

	//DEBUG
//	for (int i = 0; i < 56; i++)
//	{
//		if (!(i % 7))
//			cout << "\n";
//		cout << key_binary_ret[i];
//	}

	int subkeys_number = 17;
	int subkey_size = 28;
	int subkeys[17][56];
	int C[28], D[28];

	for (int i = 0; i < 56; i++)
		subkeys[0][i] = key_binary_ret[i];

	for(int i = 0; i < subkeys_number - 1; i++)
	{
		createSubkeys(subkeys[i], sizeof(key_binary_ret) / sizeof(key_binary_ret[0]), C, D, sizeof(C) / sizeof(C[0]), i);
		appendKeys(C, D, subkey_size, subkeys[i + 1]);
		//DEBUG
//		for(int i = 0; i < subkeys_number; i++)
//			for(int j = 0; j < 56; j++)
//				cout << 
//		for (int i = 0; i < 28; i++)
//		{
//			cout << C[i];
//		}
//		cout << endl;
//		for (int i = 0; i < 28; i++)
//		{
//			cout << D[i];
//		}
//		cout << endl;

	}

	//DEBUG
//	printArray(subkeys, 1000);

	int K[16][48];
	for(int i = 0; i < 16; i++)
	{
		permutePC(subkeys[i + 1], K[i], sizeof(K[0]) / sizeof(K[0][0]), PC_2);
	}

	//DEBUG
//	printArray2(K, 10000);

	//WARNING!!! message size 
	int message_binary_ret[64];
	permutePC(message_binary, message_binary_ret, message_size, IP);

	//DEBUG
//	for(int i = 0; i < message_size; i++)
//		cout << message_binary_ret[i];
	messageEncode(message_binary_ret, message_size, K, msg_ret);

}


void bytes2Bits(vector<BYTE> bytes, int bits[])
{
//	while(bytes.begin())
//	{
//		
//	}
	for(int i = 0; i < bytes.size(); i++)
	{
		BYTE cur = bytes[i];
		int offset = i * CHAR_BIT;

		for (int bit = 0; bit < CHAR_BIT; bit++, offset++)
		{
			bits[offset] = cur & 1;
			cur >>= 1;  // Move to next bit in array
		}
	}

	/////
//	std::bitset<numBytes * CHAR_BIT> b;
//
//	for (int i = 0; i < numBytes; ++i)
//	{
//		BYTE cur = bytes[i];
//		int offset = i * CHAR_BIT;
//
//		for (int bit = 0; bit < CHAR_BIT; ++bit)
//		{
//			b[offset] = cur & 1;
//			++offset;   // Move to next bit in b
//			cur >>= 1;  // Move to next bit in array
//		}
//	}

}

vector<BYTE> hex2Byte(string string_hex)
{
	stringstream converter;
	istringstream istringstream_hex(string_hex);
	vector<BYTE> bytes;

	string word;
	while (istringstream_hex >> word)
	{
		BYTE temp;
		converter << std::hex << word;
		converter >> temp;
		bytes.push_back(temp);
	}

	return bytes;
}

enum DesStringBase
{
	Decimal, 
	Hex,
	Binary//not implemented 
};

const char* hex_char_to_bin(char c)
{
	// TODO handle default / error
	switch (toupper(c))
	{
	case '0': return "0000";
	case '1': return "0001";
	case '2': return "0010";
	case '3': return "0011";
	case '4': return "0100";
	case '5': return "0101";
	case '6': return "0110";
	case '7': return "0111";
	case '8': return "1000";
	case '9': return "1001";
	case 'A': return "1010";
	case 'B': return "1011";
	case 'C': return "1100";
	case 'D': return "1101";
	case 'E': return "1110";
	case 'F': return "1111";
	}
}

std::string hex2Bin(const std::string& hex)
{
	// TODO use a loop from <algorithm> or smth
	std::string bin;
	for (unsigned i = 0; i != hex.length(); ++i)
		bin += hex_char_to_bin(hex[i]);
	return bin;
}

vector<int> str2Int(string& str_int)
{
	vector<int> int_vector;
	for (int i = 0; i < str_int.size(); i++)
		int_vector.push_back(str_int.c_str()[i] - '0');

	return int_vector;
}

void bin2Hex(string binary)
{
	long int longint = 0;
	for (int i = 0; i < binary.size(); i++)
		longint += (binary[binary.size() - i - 1] - 48) * pow(2, i);
	cout << setbase(16);
	cout << longint;

}

string getHexStringFromBinaryString(string sHex)
{
	string sReturn = "";
	int bit_length = 4;
	const string const bins[] = { "0000", "0001", "0010", "0011", "0100", "0101", "0110", "0111",
		"1000", "1001", "1010", "1011", "1100", "1101", "1110", "1111" };
	for (int i = 0; i < sHex.length() / bit_length; ++i)
	{
		string s = sHex.substr(bit_length * i, bit_length);

		if(s == bins[0])
			 sReturn.append("0");
		if (s == bins[1])
			 sReturn.append("1");
		if (s == bins[2])
			sReturn.append("2");
		if (s == bins[3])
			sReturn.append("3");
		if (s == bins[4])
			sReturn.append("4");
		if (s == bins[5])
			 sReturn.append("5");
		if (s == bins[6])
			 sReturn.append("6");
		if (s == bins[7])
			sReturn.append("7");
		if (s == bins[8])
			sReturn.append("8");
		if (s == bins[9])
			sReturn.append("9");
		if (s == bins[10])
			sReturn.append("A");
		if (s == bins[11])
			sReturn.append("B");
		if (s == bins[12])
			 sReturn.append("C");
		if (s == bins[13])
			 sReturn.append("D");
		if (s == bins[14])
			sReturn.append("E");
		if (s == bins[15])
			 sReturn.append("F");
		}

	return sReturn;

}


string desEncyption(string message2Encrypt, string key, DesStringBase base)
{
	//TODO implement different bases
	string str_message = hex2Bin(message2Encrypt);
	vector<int> message_binary = str2Int(str_message);
	string str_key = hex2Bin(key);
	vector<int> key_binary = str2Int(str_key);

	if(base == Decimal)
	{
		//TODO implement decimal to hex
	}

//	int message_binary[] = { 0,0,0,0, 0,0,0,1, 0,0,1,0, 0,0,1,1, 0,1,0,0, 0,1,0,1, 0,1,1,0, 0,1,1,1, 1,0,0,0, 1,0,0,1, 1,0,1,0, 1,0,1,1, 1,1,0,0, 1,1,0,1, 1,1,1,0, 1,1,1,1 };
//	int key_binary[] = { 0,0,0,1,0,0,1,1, 0,0,1,1,0,1,0,0, 0,1,0,1,0,1,1,1, 0,1,1,1,1,0,0,1, 1,0,0,1,1,0,1,1, 1,0,1,1,1,1,0,0, 1,1,0,1,1,1,1,1, 1,1,1,1,0,0,0,1 };
	int msg_ret[64];
	desEncyption(&message_binary[0], message_binary.size(), &key_binary[0], key.size(), msg_ret);
	//DEBUG
	//	for (int i = 0; i < 64; i++)
	//	{
	//		if (!(i % 8))
	//			cout << " ";
	//		cout << msg_ret[i];
	//	}

	string binary;
	for (int i = 0; i < 64; i++)
		binary.push_back(std::to_string(msg_ret[i]).c_str()[0]);
	//DEBUG
	//cout << binary;

	cout << getHexStringFromBinaryString(binary);
	
	return "NOT IMPLEMENTED";
}


//Shifts
//11110000110011001010101011110101010101100110011110001111
//11100001100110010101010111111010101011001100111100011110
//11000011001100101010101111110101010110011001111000111101
//00001100110010101010111111110101011001100111100011110101
//00110011001010101011111111000101100110011110001111010101
//11001100101010101111111100000110011001111000111101010101
//00110010101010111111110000111001100111100011110101010101
//11001010101011111111000011000110011110001111010101010110
//00101010101111111100001100111001111000111101010101011001
//01010101011111111000011001100011110001111010101010110011
//01010101111111100001100110011111000111101010101011001100
//01010111111110000110011001011100011110101010101100110011
//01011111111000011001100101010001111010101010110011001111
//01111111100001100110010101010111101010101011001100111100
//11111110000110011001010101011110101010101100110011110001
//11111000011001100101010101111010101010110011001111000111
//11110000110011001010101011110101010101100110011110001111

//K
//000110110000001011101111111111000111000001110010
//011110011010111011011001110110111100100111100101
//010101011111110010001010010000101100111110011001
//011100101010110111010110110110110011010100011101
//011111001110110000000111111010110101001110101000
//011000111010010100111110010100000111101100101111
//111011001000010010110111111101100001100010111100
//111101111000101000111010110000010011101111111011
//111000001101101111101011111011011110011110000001
//101100011111001101000111101110100100011001001111
//001000010101111111010011110111101101001110000110
//011101010111000111110101100101000110011111101001
//100101111100010111010001111110101011101001000001
//010111110100001110110111111100101110011100111010
//101111111001000110001101001111010011111100001010
//110010110011110110001011000011100001011111110101


//MESSAGE AFTER IP
//1100110000000000110011001111111111110000101010101111000010101010






int main()
{
	//TESTING
//	char msg[] = { 'a' , '\0'};
//	cout << msg << endl;
//	cout << sizeof(msg) << endl;
//	cout << sizeof(msg) * CHAR_BIT << endl;
//	
//	string str = "a";
//	cout << endl << str << endl;
//	cout << str.size() << endl;
//	cout << str.size() * CHAR_BIT << endl;
//
//	
//	int a = 0x0123456789ABCDEF;
//	stringstream stream;
//	stream << hex << a;
//	string str2 = stream.str();
//	cout << endl << str2 << endl;
//	cout << hex << str2 << endl;
//	cout << str2.size() << endl;
//	cout << str2.size() * CHAR_BIT << endl;
	//TESTING
//	int binary_int[4];
//	decimal2Binary(13, binary_int, 0);
//	reverseTab(binary_int, 4);
//	cout << binary_int[0] << binary_int[1] << binary_int[2] << binary_int[3] << endl;



//	int decimal_int = 13;
//	stringstream string_hex;
//	string_hex << decimal_int;
//	string str = string_hex.str();
//	unsigned long long value = std::stoull(str, 0, 10);
//	cout << value << endl;
//	void decimal2Binary(int decimal_int) {
//		int remainder;
//
//		if (decimal_int <= 1) {
//			cout << decimal_int;
//			return;
//		}
//
//		remainder = decimal_int % 2;
//		decimal2Binary(decimal_int >> 1);
//		cout << remainder;
//	}

//	int row[] = { 0,0,0, 1 }, column[] = { 1, 1, 0, 1 };
//	int R[4];
//	int chunk_length = 4;
//	int R_chunk[4] = { 0, 0, 0, 0 };
//	decimal2Binary(S[0][binary2Decimal(row, 4)][binary2Decimal(column, 4)], R_chunk, 0);
//	reverseTab(R_chunk, chunk_length);
//	for (int j = 0; j < chunk_length; j++)
//	{
//		R[ j] = R_chunk[j];
//	}
//
//	cout << R[0] << R[1] << R[2] << R[3] << endl;
//
//	string message = "0123456789ABCDEF";
//	string string_hex("01 23 45 67 89 AB CD EF");
////	vector<BYTE> bytes = hex2Byte(string_hex);
////	
//////	for (int i = 0; i < bytes.size(); i++)
//////	{
//////		if (!(i % 4))
//////			cout << " ";
//////		cout << bytes[i];
//////	}
////
////
////	int bits[8 * CHAR_BIT];	
////	bytes2Bits(bytes, bits);
////
////	for(int i = 0; i < 8 * CHAR_BIT / 2; i++)
////	{
////		if (!(i % 4))
////			cout << " ";
////		cout << bits[i];
////	}



	string message = "0123456789ABCDEF", key = "133457799BBCDFF1";
	int message_binary[] = { 0,0,0,0, 0,0,0,1, 0,0,1,0, 0,0,1,1, 0,1,0,0, 0,1,0,1, 0,1,1,0, 0,1,1,1, 1,0,0,0, 1,0,0,1, 1,0,1,0, 1,0,1,1, 1,1,0,0, 1,1,0,1, 1,1,1,0, 1,1,1,1};
	int key_binary[] = { 0,0,0,1,0,0,1,1, 0,0,1,1,0,1,0,0, 0,1,0,1,0,1,1,1, 0,1,1,1,1,0,0,1, 1,0,0,1,1,0,1,1, 1,0,1,1,1,1,0,0, 1,1,0,1,1,1,1,1, 1,1,1,1,0,0,0,1 };
	string cypherText = desEncyption(message, key, DesStringBase::Hex);


	//OLD
//	int key = 0x133457799BBCDFF1;
///	int* bits = get_bits(key, sizeof(key) * CHAR_BIT);
///	
///	int cntr = 0;
///	while(bits[cntr])
///	{
///		cout << bits[cntr++];
///		if (!(cntr % 8))
///			cout << " ";
///	}
//	
//	//	bytesToBitset(key);
//
////	stringstream string_hex;
///	string_hex << key;
///	string test = "0";
///	bytesToBitset<16>(string_hex.str());
///	desEncyption(message, key_binary_ret);
//	
////	char const *c_key = key_binary_ret.c_str();
///	int c_key_size = 0;
///	while (c_key[c_key_size])
///	{
///		c_key_size++;
///	}
///
///	cout << c_key_size;
//
//
////	cout << CHAR_BIT;
//
//	cout << key_binary_ret.size();
//	bitset<17 * CHAR_BIT> bits = bytesToBitset<17>(key_binary_ret.c_str());
//
//	for (int i = 0; i < bits.count(); i++)
//	{
//		if (!(i % 8))
//			cout << " ";
//		cout << bits[i];
//	}
	//OLD




	return 0;

}



////////
////////
////////
//#include <string>
//#include <bitset>
//#include <type_traits>
//
//// SFINAE for safety. Sue me for putting it in a macro for brevity on the function
//#define IS_INTEGRAL(T) typename std::enable_if< std::is_integral<T>::value >::type* = 0
//
////template<class T>
////std::string integral_to_binary_string(T byte, IS_INTEGRAL(T))
////{
////	std::bitset<sizeof(T) * CHAR_BIT> bs(byte);
////	return bs.to_string();
////}
//
//template<class T>
//std::string integral_to_binary_string(T* byte, IS_INTEGRAL(T))
//{
//	std::bitset<sizeof(T) * CHAR_BIT> map[16];  // each bitset has all 64 bits set to 0
//
//	for (int i = 0; i < 8; i++)
//	{
//		std::bitset<sizeof(unsigned char) * CHAR_BIT> bs(byte[i]);
//		cout << bs.to_string();
//		if (!(i % 2))
//			cout << " ";
//	}
//	
//	return "";
//}
//
//int main() {
//	unsigned char byte = 0x133457799BBCDFF1; // 0000 0011
////	unsigned char byte_array[] = { 0x1, 0x3, 0x3, 0x4, 0x5, 0x7, 0x7, 0x9, 0x9, 0xB, 0xB, 0xC, 0xD, 0xF, 0xF, 0x1 };
//	unsigned char byte_array[] = { 0x13, 0x34, 0x57, 0x79, 0x9B, 0xBC, 0xDF, 0xF1};
//
//	std::cout << integral_to_binary_string(byte_array);
//	//std::cin.get();
//}




//int binary2Decimal(int binary_int[])
//{
//	if (binary_int[0] == 0 && binary_int[1] == 0 && binary_int[2] == 0 && binary_int[3] == 0)
//		return ;
//	else if (binary_int[0] == && binary_int[1] == && binary_int[2] == && binary_int[3] == )
//		return ;
//	else if (binary_int[0] == && binary_int[1] == && binary_int[2] == && binary_int[3] == )
//		return ;
//	else if (binary_int[0] == && binary_int[1] == && binary_int[2] == && binary_int[3] == )
//		return ;
//	else if (binary_int[0] == && binary_int[1] == && binary_int[2] == && binary_int[3] == )
//		return ;
//	else if (binary_int[0] == && binary_int[1] == && binary_int[2] == && binary_int[3] == )
//		return ;
//	else if (binary_int[0] == && binary_int[1] == && binary_int[2] == && binary_int[3] == )
//		return ;
//	else if (binary_int[0] == && binary_int[1] == && binary_int[2] == && binary_int[3] == )
//		return ;
//	else if (binary_int[0] == && binary_int[1] == && binary_int[2] == && binary_int[3] == )
//		return ;
//	else if (binary_int[0] == && binary_int[1] == && binary_int[2] == && binary_int[3] == )
//		return ;
//	else if (binary_int[0] == && binary_int[1] == && binary_int[2] == && binary_int[3] == )
//		return ;
//	else if (binary_int[0] == && binary_int[1] == && binary_int[2] == && binary_int[3] == )
//		return ;
//	else if (binary_int[0] == && binary_int[1] == && binary_int[2] == && binary_int[3] == )
//		return ;
//	else if (binary_int[0] == && binary_int[1] == && binary_int[2] == && binary_int[3] == )
//		return ;
//	else if (binary_int[0] == && binary_int[1] == && binary_int[2] == && binary_int[3] == )
//		return ;
//	else if (binary_int[0] == && binary_int[1] == && binary_int[2] == && binary_int[3] == )
//		return ;
//	else if (binary_int[0] == && binary_int[1] == && binary_int[2] == && binary_int[3] == )
//		return;
//
//		return -1000500;
//}
//














////////////////////////////////////
//
//7 13 14 3 0 6 9 10 1 2 8 5 11 12 4 15
//13 8 11 5 6 15 0 3 4 7 2 12 1 10 14 9
//10 6 9 0 12 11 7 13 15 1 3 14 5 2 8 4
//3 15 0 6 10 1 13 8 9 4 5 11 12 7 2 14
//2 12 4 1 7 10 11 6 8 5 3 15 13 0 14 9
//14 11 2 12 4 7 13 1 5 0 15 10 3 9 8 6
//4 2 1 11 10 13 7 8 15 9 12 5 6 3 0 14
//11 8 12 7 1 14 2 13 6 15 0 9 10 4 5 3
//12 1 10 15 9 2 6 8 0 13 3 4 14 7 5 11
//10 15 4 2 7 12 9 5 6 1 13 14 0 11 3 8
//9 14 15 5 2 8 12 3 7 0 4 10 1 13 11 6
//4 3 2 12 9 5 15 10 11 14 1 7 6 0 8 13
//4 11 2 14 15 0 8 13 3 12 9 7 5 10 6 1
//13 0 11 7 4 9 1 10 14 3 5 12 2 15 8 6
//1 4 11 13 12 3 7 14 10 15 6 8 0 5 9 2
//6 11 13 8 1 4 10 7 9 5 0 15 14 2 3 12
//13 2 8 4 6 15 11 1 10 9 3 14 5 0 12 7
//1 15 13 8 10 3 7 4 12 5 6 11 0 14 9 2
//7 11 4 1 9 12 14 2 0 6 10 13 15 3 5 8
//2 1 14 7 4 10 8 13 15 12 9 0 3 5 6 11




//
//7 13 14 3 0 6 9 10 1 2 8 5 11 12 4 15
//13 8 11 5 6 15 0 3 4 7 2 12 1 10 14 9
//10 6 9 0 12 11 7 13 15 1 3 14 5 2 8 4
//3 15 0 6 10 1 13 8 9 4 5 11 12 7 2 14
//2 12 4 1 7 10 11 6 8 5 3 15 13 0 14 9
//14 11 2 12 4 7 13 1 5 0 15 10 3 9 8 6
//4 2 1 11 10 13 7 8 15 9 12 5 6 3 0 14
//11 8 12 7 1 14 2 13 6 15 0 9 10 4 5 3
//12 1 10 15 9 2 6 8 0 13 3 4 14 7 5 11
//10 15 4 2 7 12 9 5 6 1 13 14 0 11 3 8
//9 14 15 5 2 8 12 3 7 0 4 10 1 13 11 6
//4 3 2 12 9 5 15 10 11 14 1 7 6 0 8 13
//4 11 2 14 15 0 8 13 3 12 9 7 5 10 6 1
//13 0 11 7 4 9 1 10 14 3 5 12 2 15 8 6
//1 4 11 13 12 3 7 14 10 15 6 8 0 5 9 2
//6 11 13 8 1 4 10 7 9 5 0 15 14 2 3 12
//13 2 8 4 6 15 11 1 10 9 3 14 5 0 12 7
//1 15 13 8 10 3 7 4 12 5 6 11 0 14 9 2
//7 11 4 1 9 12 14 2 0 6 10 13 15 3 5 8
//2 1 14 7 4 10 8 13 15 12 9 0 3 5 6 11