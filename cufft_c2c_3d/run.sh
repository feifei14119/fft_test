rm c2c_3d 
make clean
make
./c2c_3d 100 100 100 1
nvprof --print-gpu-trace ./c2c_3d 100 100 100

