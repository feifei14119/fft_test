rm c2c_1d_len15.out
rm -rf rpl_data_* *.csv *.db

/opt/rocm/bin/hipcc -I/opt/rocm/rocfft/include/ -L/opt/rocm/rocfft/lib/ -lrocfft c2c_1d_len15.cpp -o c2c_1d_len15.out

./c2c_1d_len15.out
