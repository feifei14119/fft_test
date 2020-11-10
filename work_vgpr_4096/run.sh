rm *.out
rm -rf rpl_data_* *.csv *.db result*

/opt/rocm/bin/hipcc -I/opt/rocm/rocfft/include/ -L/opt/rocm/rocfft/lib/ -lrocfft radix_vgpr.cpp -o test.out

./test.out 4096 240 1
#rocprof -i sqtt_pmc.txt -d ./ ./test.out
#rocprof --hsa-trace --timestamp on -d ./ ./test.out
#extractkernel -i ./test.out
