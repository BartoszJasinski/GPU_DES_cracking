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


string getHexStringFromBinaryString(string sHex)
{
	string sReturn = "";
	int bit_length = 4;
	const string const bins[] = { "0000", "0001", "0010", "0011", "0100", "0101", "0110", "0111",
		"1000", "1001", "1010", "1011", "1100", "1101", "1110", "1111" };
	for (int i = 0; i < sHex.length() / bit_length; ++i)
	{
		string s = sHex.substr(bit_length * i, bit_length);

		if (s == bins[0])
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


Utils::Utils()
{
}


Utils::~Utils()
{
}
