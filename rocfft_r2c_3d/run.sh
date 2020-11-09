rm *.out
rm -rf rpl_data_* *.csv *.db result*

/opt/rocm/bin/hipcc -I/opt/rocm/rocfft/include/ -L/opt/rocm/rocfft/lib/ -lrocfft r2c_3d.cpp -o r2c_3d.out

./r2c_3d.out 100 100 100 1
rocprof -i sqtt_pmc.txt -d ./ ./r2c_3d.out
rocprof --hsa-trace --timestamp on -d ./ ./r2c_3d.out
#extractkernel -i ./r2c_3d.out
