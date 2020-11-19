rm *.out
rm -rf rpl_data_* *.csv *.db result*

/opt/rocm/bin/hipcc -I/opt/rocm/rocfft/include/ -L/opt/rocm/rocfft/lib/ -lrocfft optimize.cpp -o test.out

./test.out 112 480 1
./test.out 2187 60 1
#rocprof -i sqtt_pmc.txt -d ./ ./test.out
#rocprof --hsa-trace --timestamp on -d ./ ./test.out
#extractkernel -i ./test.out
