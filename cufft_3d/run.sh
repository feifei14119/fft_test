rm fft_3d 
make clean
make
./fft_3d
nvprof --print-gpu-trace ./fft_3d 15

