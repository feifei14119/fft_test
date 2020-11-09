rm c2c_1d 
make clean
make
./c2c_1d 100 1000 1
nvprof --print-gpu-trace ./c2c_1d 100 100

