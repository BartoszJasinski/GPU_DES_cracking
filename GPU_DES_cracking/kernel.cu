
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <iostream>
#include <bitset>
#include <sstream>
#include <stdlib.h>

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


int PC_1[56] = { 56, 48, 40, 32, 24, 16, 8, 0,
57, 49, 41, 33, 25, 17, 9, 1,
58, 50, 42, 34, 26, 18, 10, 2,
59, 51, 43, 35, 62, 54, 46, 38,
30, 22, 14, 6, 61, 53, 45, 37,
29, 21, 13, 5, 60, 52, 44, 36,
28, 20, 12, 4, 27, 19, 11, 3};


int shifts[] = { 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1 };


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
void printArray(T(&theArray)[N][M]) {
	for (int x = 0; x < N; x++) {
		for (int y = 0; y < M; y++) {
			cout << theArray[x][y];
			if (y == 27)
				cout << endl;
		}
		cout << endl;
	}
}


//bitset bytesToBitset<int numBytes>(byte *data)
//{
//	std::bitset<numBytes * CHAR_BIT> b;
//
//	for (int i = 0; i < numBytes; ++i)
//	{
//		byte cur = data[i];
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
//bitset<numBytes * CHAR_BIT>bytesToBitset(char const *data)
//{
////	char const *data = key_binary_ret.c_str();
//	bitset<numBytes * CHAR_BIT> b = *data;
//
//	for (int i = 1; i < numBytes; ++i)
//	{
//		b <<= CHAR_BIT;  // Move to next bit in array
//		b |= data[i];    // Set the lowest CHAR_BIT bits
//	}
//
//	return b;
//}
//
//

//key_binary_ret should be 64 bit long
void permutePC_1(int key_binary[], int key_binary_ret[], int key_binary_size)
{
	for (int i = 0; i < key_binary_size; i++)
		key_binary_ret[i] = key_binary[PC_1[i]];

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

void appendKeys(int leftKey[], int rightKey[], int key_size, int key_ret[])
{
	for(int i = 0; i < key_size; i++)
	{
		key_ret[i] = leftKey[i];
		key_ret[i + key_size] = rightKey[i];
	}
}

//key_binary_ret should be 64 bit long
string desEncyption(string message, int key_binary[], int key_size)
{
	int des_block_size_bytes = 8;
	int des_block_size_bits = 64;

//	cout << "omg";
	//DEBUG
//		cout << message.size();
//		cout << "\n" << message << "\n";

	if (message.size() % des_block_size_bytes)
		message.append(des_block_size_bytes - (message.size() % des_block_size_bytes), '0');//mayby another char to append  

	//DEBUG
//		cout << "\n" << message << "\n";
//		cout << message.size();

	int key_binary_ret[56];
	permutePC_1(key_binary, key_binary_ret, sizeof(key_binary_ret) / sizeof(key_binary_ret[0]));

	//DEBUG
//	for (int i = 0; i < 56; i++)
//	{
//		if (!(i % 8))
//			cout << "\n";
//		cout << key_binary_ret[i];
//	}

	int subkeys_number = 17;
	int subkey_size = 28;
	int subkeys[17][56];
	int C[28], D[28];

	for (int i = 0; i < 56; i++)
		subkeys[0][i] = key_binary_ret[i];

	for(int i = 0; i < subkeys_number; i++)
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
	printArray(subkeys);


	return "NOT IMPLEMENTED";
}



//1111000011001100101010101111
//0101010101100110011110001111
//1110000110011001010101011111
//1010101011001100111100011110
//1100001100110010101010111111
//0101010110011001111000111101
//0000110011001010101011111111
//0101011001100111100011110101
//0011001100101010101111111100
//0101100110011110001111010101
//1100110010101010111111110000
//0110011001111000111101010101
//0011001010101011111111000011
//1001100111100011110101010101
//1100101010101111111100001100
//0110011110001111010101010110
//0010101010111111110000110011
//1001111000111101010101011001
//0101010101111111100001100110
//0011110001111010101010110011
//0101010111111110000110011001
//1111000111101010101011001100
//0101011111111000011001100101
//1100011110101010101100110011
//0101111111100001100110010101
//0001111010101010110011001111
//0111111110000110011001010101
//0111101010101011001100111100
//1111111000011001100101010101
//1110101010101100110011110001
//1111100001100110010101010111
//1010101010110011001111000111
//1111000011001100101010101111
//0101010101100110011110001111

int main()
{
	string message = "0123456789ABCDEF", key = "133457799BBCDFF1";
	int key_binary[] = { 0,0,0,1,0,0,1,1, 0,0,1,1,0,1,0,0, 0,1,0,1,0,1,1,1, 0,1,1,1,1,0,0,1, 1,0,0,1,1,0,1,1, 1,0,1,1,1,1,0,0, 1,1,0,1,1,1,1,1, 1,1,1,1,0,0,0,1 };
	string cypherText = desEncyption(message, key_binary, sizeof(key_binary) / sizeof(key_binary[0]));


	//OLD
//	int key_hex = 0x133457799BBCDFF1;
///	int* bits = get_bits(key_hex, sizeof(key_hex) * CHAR_BIT);
///	
///	int cntr = 0;
///	while(bits[cntr])
///	{
///		cout << bits[cntr++];
///		if (!(cntr % 8))
///			cout << " ";
///	}
//	
//	//	bytesToBitset(key_hex);
//
////	stringstream ss;
///	ss << key_hex;
///	string test = "0";
///	bytesToBitset<16>(ss.str());
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