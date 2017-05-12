#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <vector>
#include <iomanip>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.cuh"
#include "Utils.h"
#include "BinaryUtils.cuh"
#include "Arrays.h"

typedef unsigned char BYTE;

using namespace std;

__device__ int key_cracked = 0;

__constant__ int d_PC_1[56], d_shifts[16], d_PC_2[48], d_IP[64], d_E[48], d_S[8][4][16], d_P[32], d_IP_1[64];

__device__ void permutePC(int key_binary[], int key_binary_ret[], int key_binary_size, const int PC[])
{
	for (int i = 0; i < key_binary_size; i++)
		key_binary_ret[i] = key_binary[PC[i]];

}

//C and D should have 28 array memebers PROBABLY!!!
__device__ void createSubkeys(int key[], const int key_size, int C[], int D[], int CD_size, int run_number)
{
	const int size = key_size / 2;
	int tmp_C[28], tmp_D[28];
	for (int i = 0; i < key_size / 2; i++)
	{
		tmp_C[i] = key[i];
		tmp_D[i] = key[i + CD_size];
	}

	for (int i = 0; i < CD_size; i++)
	{
		C[i] = tmp_C[(i + d_shifts[run_number]) % CD_size];
		D[i] = tmp_D[(i + d_shifts[run_number]) % CD_size];
	}

}


__host__ __device__ void decimal2Binary(int decimal_int, int binary_int[], int run_number)
{
	if (decimal_int <= 1) {
		binary_int[run_number] = decimal_int;
		return;
	}

	int remainder = decimal_int % 2;
	decimal2Binary(decimal_int >> 1, binary_int, run_number + 1);
	binary_int[run_number] = remainder;
}


__device__ void reverseTab(int tab[], int tab_length)
{
	for (int i = 0; i < tab_length / 2; i++)
	{
		int tmp = tab[i];
		tab[i] = tab[tab_length - i - 1];
		tab[tab_length - i - 1] = tmp;

	}
}


__device__ void appendKeys(int leftKey[], int rightKey[], int key_size, int key_ret[])
{
	for (int i = 0; i < key_size; i++)
	{
		key_ret[i] = leftKey[i];
		key_ret[i + key_size] = rightKey[i];
	}
}


__device__ void expand(int R[], int tab_ret[], const int E[], int E_size)
{

	for (int i = 0; i < E_size; i++)
		tab_ret[i] = R[E[i]];
}


__device__ void xorArray(int first_tab[], int second_tab[], int tab_size, int tab_ret[])
{
	for (int i = 0; i < tab_size; i++)
		tab_ret[i] = (int)(!first_tab[i] != !second_tab[i]);

}


__device__ long long binary2Decimal(int binary_int[], int tab_length)
{
	long long dec = 0;

	for (int i = 0; i < tab_length; ++i)
	{
		int bin = binary_int[i];
		if (bin) dec = dec * 2 + 1;
		else dec *= 2;

	}

	return dec;
}


__device__ void f(int R[], int K[], int ret_tab[])
{
	int R_expanded[48];
	expand(R, R_expanded, d_E, 48);
	//DEBUG
	//	for (int i = 0; i < 48; i++)
	//	{
	//		if (!(i % 6))
	//			cout << " ";
	//		cout << R_expanded[i];
	//	}
	//	cout << endl << endl << endl;

	int xored[48];
	xorArray(K, R_expanded, 48, xored);
	//DEBUG
	//	for(int i = 0; i < 48; i++)
	//	{
	//		if (!(i % 6))
	//			cout << " ";
	//		cout << xored[i];
	//	}
	//	cout << endl << endl << endl;


	for (int i = 0; i < 8; i++)
	{
		int row[4] = { 0, 0, 0, 0 }, column[4] = { 0, 0, 0, 0 };
		row[3] = xored[6 * i + 5];
		row[2] = xored[6 * i];
		column[0] = xored[6 * i + 1];
		column[1] = xored[6 * i + 2];
		column[2] = xored[6 * i + 3];
		column[3] = xored[6 * i + 4];

		int chunk_length = 4;
		int R_chunk[4] = { 0, 0, 0, 0 };
		decimal2Binary(d_S[i][binary2Decimal(row, 4)][binary2Decimal(column, 4)], R_chunk, 0);
		reverseTab(R_chunk, chunk_length);
		for (int j = 0; j < chunk_length; j++)
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

	permutePC(R, ret_tab, 32, d_P);
	//DEBUG
	//	for (int i = 0; i < 32; i++)
	//	{
	//		if (!(i % 4))
	//			cout << " ";
	//		cout << ret_tab[i];
	//	}
	//	cout << endl << endl << endl;

}


__device__ void reverse(int L[], int R[], int tab_length, int ret_tab[])
{
	for (int i = 0; i < tab_length; i++)
	{
		ret_tab[i] = R[i];
		ret_tab[i + tab_length] = L[i];
	}
}


__device__ void messageEncode(int message_binary[], int message_size, int K[][48], int msg_ret[])
{
	int L[32], R[32];
	for (int i = 0; i < message_size / 2; i++)
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

	for (int i = 0; i < 16; i++)
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

		xorArray(prev_L, tmp_f, 32, R);

		//DEBUG
		//		for(int i = 0; i < 32; i++)
		//		{
		//			if (!(i % 4))
		//				cout << " ";
		//			cout << R[i];
		//		}
		//		cout << endl << endl << endl;

		//przepisanie R i L do prev_R i prev_L
		for (int j = 0; j < message_size / 2; j++)
		{
			prev_L[j] = L[j];
			prev_R[j] = R[j];
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

	permutePC(msg, msg_ret, 64, d_IP_1);
	//DEBUG
	//	for(int i = 0; i < 64; i++)
	//	{
	//		if (!(i % 8))
	//			cout << " ";
	//		cout << msg_ret[i];
	//	}	

}


//key_binary_ret should be 64 bit long
__device__ void desEncryption(int message_binary[], int message_size, int key_binary[], int key_size, int msg_ret[])
{
	int des_block_size_bytes = 8;
	int des_block_size_bits = 64;

	//DEBUG
	//	printf("\n%s\n", "__device__ desEncryptionForDataBlock ");
	//	printf("%s\n", "message_binary");
	//	for (int i = 0; i < message_size; ++i)
	//	{
	//		printf("%i", message_binary[i]);
	//	}
	//		cout << message.size();
	//		cout << "\n" << message << "\n";
	//if (message.size() * CHAR_BIT != des_block_size_bits)
	//		cout << message.size() * CHAR_BIT;


	if (message_size % des_block_size_bytes)
	{
		//int tmp_message_binary[message_size + des_block_size_bytes - (message_size % des_block_size_bytes)]
		printf("%s\n", "KICIA");
		//	message_binary.append(des_block_size_bytes - (message.size() % des_block_size_bytes), '0');//mayby another char to append  
	}

	//OLD Verwsion with message as string 
	//	if (message.size() % des_block_size_bytes)
	//		message.append(des_block_size_bytes - (message.size() % des_block_size_bytes), '0');//mayby another char to append  

	//DEBUG
	//		cout << "\n" << message << "\n";
	//		cout << message.size();

	int key_binary_ret[56];
	permutePC(key_binary, key_binary_ret, sizeof(key_binary_ret) / sizeof(key_binary_ret[0]), d_PC_1);

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

	for (int i = 0; i < subkeys_number - 1; i++)
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
	for (int i = 0; i < 16; i++)
	{
		permutePC(subkeys[i + 1], K[i], sizeof(K[0]) / sizeof(K[0][0]), d_PC_2);
	}

	//DEBUG
	//	printArray2(K, 10000);

	//WARNING!!! message size 
	int message_binary_ret[64];
	permutePC(message_binary, message_binary_ret, message_size, d_IP);

	//DEBUG
	//	for(int i = 0; i < message_size; i++)
	//		cout << message_binary_ret[i];
	messageEncode(message_binary_ret, message_size, K, msg_ret);

}


const char* hexChar2Bin(char c)
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
		default:
			return "ERROR_hexChar2Bin";
	}
}


string hex2Bin(const string& hex)
{
	// TODO use a loop from <algorithm> or smth
	string bin;
	for (unsigned i = 0; i != hex.length(); ++i)
		bin += hexChar2Bin(hex[i]);
	return bin;
}


void str2Int(string& str_int, int ret_int[], int ret_int_size)
{
	for (int i = 0; i < ret_int_size; i++)
		ret_int[i] = (str_int.c_str()[i] - '0');

}


__host__ __device__ void consecutiveKeyGenerator(unsigned long long &present_key, int next_key_binary[], int next_key_binary_size)
{
	for (int i = 0; i < next_key_binary_size; i++)
		next_key_binary[i] = 0;
	decimal2Binary(present_key, next_key_binary, 0);
}


__host__ __device__ bool compareArrays(int message[], int cyphertext[])
{
	for (int i = 0; i < 64; i++)
	{
		if (message[i] != cyphertext[i])
			return false;
	}

	return true;
}


__global__
void crackDes(int message_binary[], int cyphertext_binary[], int message_binary_size, unsigned long long computation_size)
{
	//printf("%s\n", "__global__ crackDes");



	//DEBUG
	//	for (int i = 0; i < possible_key_binary_size; ++i)
	//	{
	//		printf("%i", possible_key_binary[i]);
	//	}

	int msg_ret[64];

	//	printf("%s\n", "BEFORE desEncryptionForDataBlock");

	int possible_key_binary_size = 56;
	int possible_key_binary[56];
	unsigned long long present_key = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long long last_key = present_key + computation_size;

	//unsigned long long temp = present_key + 2147483648;
	for (unsigned long long i = present_key; i < last_key; i++)
	{
		if (key_cracked == 1)
		{
			//		printf("%i", key_cracked);
			return;
		}
		consecutiveKeyGenerator(i, possible_key_binary, possible_key_binary_size);
		//	printf("%s\n", "BEFORE desEncryptionForDataBlock");
		desEncryption(message_binary, message_binary_size, possible_key_binary, 16, msg_ret);
		//	printf("%s\n", "AFTER desEncryptionForDataBlock");

		if (compareArrays(msg_ret, cyphertext_binary))
		{
			key_cracked = 1;
			printf("%s", "USED KEY IS: ");
			for (int i = 0; i < possible_key_binary_size; i++)
				printf("%i", possible_key_binary[i]);
			printf("\n");
		}
	}
}


__host__
void crackDes(string message, string cyphertext)
{
	string str_message = hex2Bin(message);
	int h_message_binary_size = 64;
	int h_message_binary[64];
	str2Int(str_message, h_message_binary, h_message_binary_size);

	string str_cyphertext = hex2Bin(cyphertext);
	int h_cyphertext_binary_size = 64;
	int h_cyphertext_binary[64];
	str2Int(str_cyphertext, h_cyphertext_binary, h_cyphertext_binary_size);

	int* d_message_binary = 0;
	cudaMalloc((void**)&d_message_binary, h_message_binary_size * sizeof(int));
	cudaMemcpy(d_message_binary, h_message_binary, h_message_binary_size * sizeof(int), cudaMemcpyHostToDevice);

	int* d_cyphertext_binary = 0;
	cudaMalloc((void**)&d_cyphertext_binary, h_cyphertext_binary_size * sizeof(int));
	cudaMemcpy(d_cyphertext_binary, h_cyphertext_binary, h_cyphertext_binary_size * sizeof(int), cudaMemcpyHostToDevice);

	const int threads_per_block = 512;//FERMI //1024; //2^10
	const int nbr_of_block_in_one_dim = 8192; //2 ^ 13;
	const int test_nbr_of_block = 32768; //2 ^ 15
	unsigned long long computation_size = pow(2, 47) / (nbr_of_block_in_one_dim);
//	printf("%s\n", "__host__ crackDes BEFORE __device__ crackDes");
	crackDes<<<nbr_of_block_in_one_dim, threads_per_block>>>(d_message_binary, d_cyphertext_binary, h_message_binary_size, computation_size);
//	printf("%s\n", "__host__ crackDes AFTER __device__ crackDes");
	//DEBUG
	//	for (int i = 0; i < 64; i++)
	//	{
	//		if (!(i % 8))
	//			cout << " ";
	//		cout << msg_ret[i];
	//	}

	//	string binary;
	//	for (int i = 0; i < 64; i++)
	//		binary.push_back(std::to_string(msg_ret[i]).c_str()[0]);
	//DEBUG
	//cout << binary;

}


__global__
void desEncryption(int message_binary[], int key_binary[], int message_binary_size, int msg_ret[])
{
	//DEBUG
	//	printf("%s\n", "before DEBUG __global__ desEncryptionForDataBlock MESSAGE_BINARY");
	//	for (int i = 0; i < message_binary_size; ++i)
	//	{
	//		printf("%i", message_binary[i]);
	//	}
	//	printf("%s\n", "after DEBUG __global__ desEncryptionForDataBlock MESSAGE_BINARY");

	//int msg_ret[64];
	//	printf("%s\n", "BEFORE desEncryptionForDataBlock");							14 should be here
	desEncryption(message_binary, message_binary_size, key_binary, 16, msg_ret);
	//	printf("%s\n", "before DEBUG __global__ desEncryptionForDataBlock MSG_RET");
	//	for (int i = 0; i < 64; ++i)
	//	{
	//		printf("%i", 123123123);
	//	}
	//	printf("%s\n", "after DEBUG __global__ desEncryptionForDataBlock MSG_RET");

}


__host__
string desEncryptionForDataBlock(string message, string key)
{
	string str_message = hex2Bin(message);
	int h_message_binary_size = 64;
	int h_message_binary[64];
	str2Int(str_message, h_message_binary, h_message_binary_size);

	string str_key = hex2Bin(key);
	int h_key_binary_size = 56;
	int h_key_binary[56];
	str2Int(str_key, h_key_binary, h_key_binary_size);

	int* d_message_binary = 0;
	cudaMalloc((void**)&d_message_binary, h_message_binary_size * sizeof(int));
	cudaMemcpy(d_message_binary, h_message_binary, h_message_binary_size * sizeof(int), cudaMemcpyHostToDevice);

	int* d_key_binary = 0;
	cudaMalloc((void**)&d_key_binary, h_key_binary_size * sizeof(int));
	cudaMemcpy(d_key_binary, h_key_binary, h_key_binary_size * sizeof(int), cudaMemcpyHostToDevice);

	int* d_msg_ret;
	cudaMalloc((void**)&d_msg_ret, 64 * sizeof(int));

	//DEBUG
	//	printf("%s\n", "before DEBUG __host__ desEncryptionForDataBlock");
	//	for (int i = 0; i < 64; ++i)
	//	{
	//		printf("%i", h_message_binary[i]);
	//	}
	//	printf("%s\n", "after DEBUG __host__ desEncryptionForDataBlock");

	desEncryption<<<1, 1 >>>(d_message_binary, d_key_binary, 64, d_msg_ret);

	cudaDeviceSynchronize();

	int* h_msg_ret = (int*)malloc(64 * sizeof(int));
	cudaMemcpy(h_msg_ret, d_msg_ret, 64 * sizeof(int), cudaMemcpyDeviceToHost);
	//DEBUG
	//	printf("\n%s\n", "before DEBUG __host__ desEncryptionForDataBlock H_MSG_RET");
	//	for (int i = 0; i < 64; ++i)
	//	{
	//		printf("%i", h_msg_ret[i]);
	//	}
	//	printf("%s\n", "after DEBUG __host__ desEncryptionForDataBlock H_MSG_RET");


	string binary;
	for (int i = 0; i < 64; i++)
		binary.push_back(std::to_string(h_msg_ret[i]).c_str()[0]);
	//DEBUG
	//cout << binary;

	return getHexStringFromBinaryString(binary);

}


__host__
string desEncryption(string message, string key)
{
	int block_size = 16;
	string encryptedMessage = "";
	for (int i = 0; i < message.size() / block_size; ++i)
		encryptedMessage += desEncryptionForDataBlock(message.substr(i * block_size, block_size), key);

	return encryptedMessage;
}


void initArrays()
{
	cudaMemcpyToSymbol(d_PC_1, PC_1, PC_1_size * sizeof(int));
	cudaMemcpyToSymbol(d_shifts, shifts, shifts_size * sizeof(int));
	cudaMemcpyToSymbol(d_PC_2, PC_2, PC_2_size * sizeof(int));
	cudaMemcpyToSymbol(d_IP, IP, IP_size * sizeof(int));
	cudaMemcpyToSymbol(d_E, E, E_size * sizeof(int));
	cudaMemcpyToSymbol(d_P, P, P_size * sizeof(int));
	cudaMemcpyToSymbol(d_S, S, S_size_1 * S_size_2 * S_size_3 * sizeof(int));
	cudaMemcpyToSymbol(d_IP_1, IP_1, IP_1_size * sizeof(int));

}


void resizeGPUHeap()
{
	size_t size_heap, size_stack;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 20000000 * sizeof(double));
	cudaDeviceSetLimit(cudaLimitStackSize, 12928);
	cudaDeviceGetLimit(&size_heap, cudaLimitMallocHeapSize);
	cudaDeviceGetLimit(&size_stack, cudaLimitStackSize);
	printf("Heap size found to be %d; Stack size found to be %d\n", (int)size_heap, (int)size_stack);

}
