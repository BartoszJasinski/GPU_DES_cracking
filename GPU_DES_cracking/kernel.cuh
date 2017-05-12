#pragma once
#include <string>
#include "device_launch_parameters.h"

__host__ __device__ void decimal2Binary(int decimal_int, int binary_int[], int run_number);

__device__ void permutePC(int key_binary[], int key_binary_ret[], int key_binary_size, const int PC[]);

__device__ void createSubkeys(int key[], const int key_size, int C[], int D[], int CD_size, int run_number);

__device__ void reverseTab(int tab[], int tab_length);

__device__ void appendKeys(int leftKey[], int rightKey[], int key_size, int key_ret[]);

__device__ void expand(int R[], int tab_ret[], const int E[], int E_size);

__device__ void xorArray(int first_tab[], int second_tab[], int tab_size, int tab_ret[]);

__device__ long long binary2Decimal(int binary_int[], int tab_length);

__device__ void f(int R[], int K[], int ret_tab[]);

__device__ void reverse(int L[], int R[], int tab_length, int ret_tab[]);

__device__ void messageEncode(int message_binary[], int message_size, int K[][48], int msg_ret[]);

__device__ void desEncryption(int message_binary[], int message_size, int key_binary[], int key_size, int msg_ret[]);

const char* hexChar2Bin(char c);

std::string hex2Bin(const std::string& hex);

void str2Int(std::string& str_int, int ret_int[], int ret_int_size);

std::string getHexStringFromBinaryString(std::string sHex);

__host__ __device__ void consecutiveKeyGenerator(unsigned long long &present_key, int next_key_binary[], int next_key_binary_size);

__host__ __device__ bool compareArrays(int message[], int cyphertext[]);

__global__
void crackDes(int message_binary[], int cyphertext_binary[], int message_binary_size, unsigned long long computation_size);

__host__
void crackDes(std::string message, std::string cyphertext);

__global__
void desEncryption(int message_binary[], int key_binary[], int message_binary_size, int msg_ret[]);

__host__
std::string desEncryptionForDataBlock(std::string message, std::string key);

__host__
std::string desEncryption(std::string message, std::string key);

void initArrays();

void resizeGPUHeap();

