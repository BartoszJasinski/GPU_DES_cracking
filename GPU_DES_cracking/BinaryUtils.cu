#include "BinaryUtils.cuh"

//
//__host__ __device__ void decimal2Binary(int decimal_int, int binary_int[], int run_number)
//{
//	if (decimal_int <= 1) {
//		binary_int[run_number] = decimal_int;
//		return;
//	}
//
//	int remainder = decimal_int % 2;
//	decimal2Binary(decimal_int >> 1, binary_int, run_number + 1);
//	binary_int[run_number] = remainder;
//}



BinaryUtils::BinaryUtils()
{
}


BinaryUtils::~BinaryUtils()
{
}
