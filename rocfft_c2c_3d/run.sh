rm *.out
rm -rf rpl_data_* *.csv *.db result*

/opt/rocm/bin/hipcc -I/opt/rocm/rocfft/include/ -L/opt/rocm/rocfft/lib/ -lrocfft c2c_3d.cpp -o c2c_3d.out

./c2c_3d.out 100 100 100 1
rocprof -i sqtt_pmc.txt -d ./ ./c2c_3d.out
rocprof --hsa-trace --timestamp on -d ./ ./c2c_3d.out
#extractkernel -i ./c2c_3d.out
