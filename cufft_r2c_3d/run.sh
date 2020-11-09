rm r2c_3d 
make clean
make
./r2c_3d 100 100 100 1
nvprof --print-gpu-trace ./r2c_3d 100 100 100

