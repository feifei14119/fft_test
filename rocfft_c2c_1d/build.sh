rm *.out
rm -rf rpl_data_* *.csv *.db result*

/opt/rocm/bin/hipcc -I/opt/rocm/rocfft/include/ -L/opt/rocm/rocfft/lib/ -lrocfft c2c_1d.cpp -o c2c_1d.out

./c2c_1d.out
#rocprof -i sqtt_pmc.txt -d ./ ./c2c_1d.out
#rocprof --hsa-trace --timestamp on -d ./ ./c2c_1d.out
