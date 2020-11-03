rm c2c_1d 
make clean
make
./c2c_1d
nvprof --print-gpu-trace ./c2c_1d 15

