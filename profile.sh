nvcc main.cu -o main -O3 -w -arch=sm_75
nsys profile -f true -o "report_main_local" ./main roadNet-CA.mtx 128 128
nsys-ui report_main_local.nsys-rep 