#include "Utils.h"
#include <iostream>

/*void fun()
{
	for (int i = 0; i < 56; i++)
	{
		std::cout << d_PC_1[i] - 1 << ", ";

		if (!(i % 8))
			std::cout << "\n";
	}
}*/

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


template< typename T, size_t N, size_t M >
void printArray2(T(&theArray)[N][M], int char_endl_nbr) {
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


Utils::Utils()
{
}


Utils::~Utils()
{
}
