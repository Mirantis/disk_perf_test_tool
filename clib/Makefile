libwally.so: interpolate.o Makefile
	g++ interpolate.o -shared -o libwally.so

interpolate.o: interpolate.cpp
	g++ -c -Wall -std=c++11 -fPIC -O3 interpolate.cpp -o interpolate.o

