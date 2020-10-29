rm r2c_3d.out
rm -rf rpl_data_* *.csv *.db

/opt/rocm/bin/hipcc -I/opt/rocm/rocfft/include/ -L/opt/rocm/rocfft/lib/ -lrocfft r2c_3d.cpp -o r2c_3d.out

./r2c_3d.out

