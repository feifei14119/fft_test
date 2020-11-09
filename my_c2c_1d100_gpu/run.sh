rm -rf rpl_data_*
rm *.out *.bundle *.hsaco *.isa  *.cvs

/opt/rocm/bin/hipcc -I/opt/rocm/rocfft/include/ -L/opt/rocm/rocfft/lib/ -lrocfft gpu_1d100.cpp -o gpu_1d100.out

./gpu_1d100.out 1 1
rocprof -i ./sqtt_pmc.txt -d ./ ./gpu_1d100.out
rocprof --hsa-trace --timestamp on -d ./ ./gpu_1d100.out
extractkernel -i gpu_1d100.out
