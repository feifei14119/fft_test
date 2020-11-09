#hipcc -I/opt/rocm/include -L/opt/rocm/lib -lrocfft real2complex_3d.cpp -o real2complex_3d.out
#hipcc -I/home/feifei/rocFFT/build/include/ -L/home/feifei/rocFFT/build/library/src/ -lrocfft real2complex_3d.cpp -o real2complex_3d.out

cd /feifei/rocFFT/build
make -j16

cd /feifei/projects/fft
cp /feifei/rocFFT/build/library/src/librocfft.so* /opt/rocm/rocfft/lib/
cp /feifei/rocFFT/build/library/src/device/librocfft-device.so* /opt/rocm/rocfft/lib/

rm r2c_3d_test.out
rm -rf rpl_data_* *.csv *.db
/opt/rocm/bin/hipcc -I/feifei/rocFFT/build/include/ -L/feifei/rocFFT/build/library/src/ -lrocfft r2c_3d_test.cpp -o r2c_3d_test.out

./r2c_3d_test.out
#rocprof -i sqtt_pmc.txt -d ./ ./real2complex_3d.out
#rocprof --hsa-trace --timestamp on -d ./ ./real2complex_3d.out
