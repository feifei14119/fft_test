rm fft_1d 
make clean
make
./fft_1d
nvprof --print-gpu-trace ./fft_1d 15

