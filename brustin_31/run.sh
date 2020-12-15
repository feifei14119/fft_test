rm *.out
rm -rf rpl_data_* *.csv *.db result*

#/opt/rocm/bin/hipcc -I/opt/rocm/rocfft/include/ -L/opt/rocm/rocfft/lib/ -lrocfft c2c_1d_31.cpp -o test.out
/opt/rocm/bin/hipcc -I/opt/rocm/rocfft/include/ -L/opt/rocm/rocfft/lib/ -lrocfft my_c2c_1d_31.cpp -o test.out

#/opt/rocm/bin/hipcc -I/opt/rocm/rocfft/include/ -L/opt/rocm/rocfft/lib/ -lrocfft c2c_3d_31.cpp -o test.out

#./test.out 31 20 1
rocprof --hip-trace ./test.out
./test.out
#rocprof -i sqtt_pmc.txt -d ./ ./test.out
#rocprof --hsa-trace --timestamp on -d ./ ./test.out
#extractkernel -i ./test.out
