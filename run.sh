g++ main-cpu.cpp -o main-cpu -O3 -w
./main-cpu roadNet-CA.mtx 128 128

nvcc main.cu -o main -O3 -w -arch=sm_75
./main roadNet-CA.mtx 128 128
./main tests/standard.txt 128 128
./main tests/standard3.txt 128 128
./main tests/standard6.txt 128 128