rm r2c_1d 
make clean
make
./r2c_1d 100 100 1
nvprof --print-gpu-trace ./r2c_1d 100 100

