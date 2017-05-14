#include "Main.h"
#include "kernel.cuh"
#include "Utils.h"

#include <string>
#include <iostream>
#include <cuda_runtime_api.h>

using namespace std;


int main()
{
	//cudaSetDevice(3); //uncomment when using gpunode1
	resizeGPUHeap();
	initArrays();
	//message and key has to be in HEX
	string message = "01", key = "0B000000000000";
	string ct = desEncryption(message, key);

	cout << ct << "\n";
	crackDes(message, ct.c_str());
	//	cout << "MAIN: AFTER crackDes";
	cudaDeviceSynchronize();

	cout << endl << "END main" << endl;

	return 0;
}


Main::Main()
{
}


Main::~Main()
{
}
