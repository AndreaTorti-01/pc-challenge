nvcc main.cu -o main -O3 -w
./main
nsys profile -f true -o "report_main_local" ./main
nsys-ui report_main_local.nsys-rep 