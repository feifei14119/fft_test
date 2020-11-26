# nohup python3 perf_test.py > test.log 2>&1 &
# jobs
# kill %jobnum
# fg %jobnum
# bg %jobnum

import subprocess
import os, re 
import xlwt
import time

test_length = [112, 512, 2187, 3125, 4096];
test_batch = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096]
	
workbook = 0;
worksheet = 0;
xl_row_cnt = 0;
init_col = 2;
init_row = 2;

def execCmd(cmd):  
	r = os.popen(cmd)  
	text = r.read()  
	r.close()  
	return text 
	
def test1DrocOneParam(length, batch):
	cmd = "./rocFFT/build/clients/staging/rocfft-rider -o --length {0} -b {1}".format(length, batch)
	print(cmd)
	result = execCmd(cmd)
	result = re.split("\n", result)
	time = float(0)
	for line in result:
		idx1 = line.rfind("gpu time: ")
		idx2 = line.rfind(" ms")
		if(idx1 != -1):
			print(line)
			line = line[idx1 + len("gpu time: "):idx2-1]
			time = float(line)
	return time
def test1DffOneParam(length, batch):
	cmd = "./ffFFT/build/clients/staging/rocfft-rider -o --length {0} -b {1}".format(length, batch)
	print(cmd)
	result = execCmd(cmd)
	result = re.split("\n", result)
	time = float(0)
	for line in result:
		idx1 = line.rfind("gpu time: ")
		idx2 = line.rfind(" ms")
		if(idx1 != -1):
			print(line)
			line = line[idx1 + len("gpu time: "):idx2-1]
			time = float(line)
	return time
def test3DrocOneParam(length):
	cmd = "./rocFFT/build/clients/staging/rocfft-rider -o --length {0} {0} {0}".format(length, length, length)
	print(cmd)
	result = execCmd(cmd)
	result = re.split("\n", result)
	time = float(0)
	for line in result:
		idx1 = line.rfind("gpu time: ")
		idx2 = line.rfind(" ms")
		if(idx1 != -1):
			print(line)
			line = line[idx1 + len("gpu time: "):idx2-1]
			time = float(line)
	return time
def test3DffOneParam(length):
	cmd = "./ffFFT/build/clients/staging/rocfft-rider -o --length {0} {0} {0}".format(length, length, length)
	print(cmd)
	result = execCmd(cmd)
	result = re.split("\n", result)
	time = float(0)
	for line in result:
		idx1 = line.rfind("gpu time: ")
		idx2 = line.rfind(" ms")
		if(idx1 != -1):
			print(line)
			line = line[idx1 + len("gpu time: "):idx2-1]
			time = float(line)
	return time

def test1D():
	global worksheet; 
	global xl_row_cnt;
	xl_col_cnt = init_col;
	xl_row_cnt = init_row;
	worksheet.write(xl_row_cnt, xl_col_cnt, label = 'length');  xl_col_cnt = xl_col_cnt + 1;
	worksheet.write(xl_row_cnt, xl_col_cnt, label = 'batch');   xl_col_cnt = xl_col_cnt + 1;
	worksheet.write(xl_row_cnt, xl_col_cnt, label = 'origin');  xl_col_cnt = xl_col_cnt + 1;
	worksheet.write(xl_row_cnt, xl_col_cnt, label = 'vgpr');    xl_col_cnt = xl_col_cnt + 1;
	worksheet.write(xl_row_cnt, xl_col_cnt, label = 'diff');    xl_col_cnt = xl_col_cnt + 1;
	
	for l in test_length:
		for b in test_batch:
			t1 = test1DrocOneParam(l, b);
			t2 = test1DffOneParam(l, b);
			xl_col_cnt = init_col;
			xl_row_cnt = xl_row_cnt + 1;
			worksheet.write(xl_row_cnt, xl_col_cnt, label = l);  	xl_col_cnt = xl_col_cnt + 1;
			worksheet.write(xl_row_cnt, xl_col_cnt, label = b);  	xl_col_cnt = xl_col_cnt + 1;
			worksheet.write(xl_row_cnt, xl_col_cnt, label = t1);	xl_col_cnt = xl_col_cnt + 1;
			worksheet.write(xl_row_cnt, xl_col_cnt, label = t2);	xl_col_cnt = xl_col_cnt + 1;
			worksheet.write(xl_row_cnt, xl_col_cnt, label = t1 - t2);	xl_col_cnt = xl_col_cnt + 1;
			print("fffft  1d length = {0}, batch = {1}, origin = {2}(ms), vgpr = {3}(ms)".format(l, b, t1, t2));
def test1D2(length):
	global worksheet; 
	global xl_row_cnt;
	xl_col_cnt = init_col;
	xl_row_cnt = init_row;
	worksheet.write(xl_row_cnt, xl_col_cnt, label = 'length');  xl_col_cnt = xl_col_cnt + 1;
	worksheet.write(xl_row_cnt, xl_col_cnt, label = 'batch');   xl_col_cnt = xl_col_cnt + 1;
	worksheet.write(xl_row_cnt, xl_col_cnt, label = 'origin');  xl_col_cnt = xl_col_cnt + 1;
	worksheet.write(xl_row_cnt, xl_col_cnt, label = 'vgpr');    xl_col_cnt = xl_col_cnt + 1;
	worksheet.write(xl_row_cnt, xl_col_cnt, label = 'diff');    xl_col_cnt = xl_col_cnt + 1;
	
	l = length;
	for b in range(1000,9100,100):
		t1 = test1DrocOneParam(l, b);
		t2 = test1DffOneParam(l, b);
		xl_col_cnt = init_col;
		xl_row_cnt = xl_row_cnt + 1;
		worksheet.write(xl_row_cnt, xl_col_cnt, label = l);  	xl_col_cnt = xl_col_cnt + 1;
		worksheet.write(xl_row_cnt, xl_col_cnt, label = b);  	xl_col_cnt = xl_col_cnt + 1;
		worksheet.write(xl_row_cnt, xl_col_cnt, label = t1);	xl_col_cnt = xl_col_cnt + 1;
		worksheet.write(xl_row_cnt, xl_col_cnt, label = t2);	xl_col_cnt = xl_col_cnt + 1;
		worksheet.write(xl_row_cnt, xl_col_cnt, label = t1 - t2);	xl_col_cnt = xl_col_cnt + 1;
		print("fffft  1d length = {0}, batch = {1}, origin = {2}(ms), vgpr = {3}(ms)".format(l, b, t1, t2));
def test3D():
	global worksheet; 
	global xl_row_cnt;
	xl_col_cnt = init_col;
	xl_row_cnt = init_row;
	worksheet.write(xl_row_cnt, xl_col_cnt, label = 'length');  xl_col_cnt = xl_col_cnt + 1;
	worksheet.write(xl_row_cnt, xl_col_cnt, label = 'origin');  xl_col_cnt = xl_col_cnt + 1;
	worksheet.write(xl_row_cnt, xl_col_cnt, label = 'vgpr');    xl_col_cnt = xl_col_cnt + 1;
	worksheet.write(xl_row_cnt, xl_col_cnt, label = 'diff');    xl_col_cnt = xl_col_cnt + 1;
	
	for l in test_length:
		t1 = test3DrocOneParam(l)
		t2 = test3DffOneParam(l)
		xl_col_cnt = init_col;
		xl_row_cnt = xl_row_cnt + 1;
		worksheet.write(xl_row_cnt, xl_col_cnt, label = l);  	xl_col_cnt = xl_col_cnt + 1;
		worksheet.write(xl_row_cnt, xl_col_cnt, label = t1);	xl_col_cnt = xl_col_cnt + 1;
		worksheet.write(xl_row_cnt, xl_col_cnt, label = t2);	xl_col_cnt = xl_col_cnt + 1;
		worksheet.write(xl_row_cnt, xl_col_cnt, label = t1 - t2);	xl_col_cnt = xl_col_cnt + 1;
		print("fft 3d length = {0}, origin = {1}(ms), vgpr = {2}(ms)".format(l, t1, t2));
	
def testAll():
	global workbook; 
	global worksheet; 
	global xl_row_cnt;
	workbook = xlwt.Workbook(encoding = 'utf-8');
	
	print("------------------------------------------------------------------------");
	worksheet = workbook.add_sheet('fft 1d'); test1D(); workbook.save('fft_perf.xls');
	print("------------------------------------------------------------------------");
	worksheet = workbook.add_sheet('112 1d batch');  test1D2(112);  workbook.save('fft_perf.xls');
	worksheet = workbook.add_sheet('512 1d batch');  test1D2(512);  workbook.save('fft_perf.xls');
	worksheet = workbook.add_sheet('2187 1d batch'); test1D2(2187); workbook.save('fft_perf.xls');
	worksheet = workbook.add_sheet('3125 1d batch'); test1D2(3125); workbook.save('fft_perf.xls');
	worksheet = workbook.add_sheet('4096 1d batch'); test1D2(4096); workbook.save('fft_perf.xls');
	print("------------------------------------------------------------------------");
	worksheet = workbook.add_sheet('fft 3d'); test3D(); workbook.save('fft_perf.xls');
	print("------------------------------------------------------------------------");
	
	workbook.save('fft_perf.xls')
	
if __name__ == '__main__': 
	testAll()
	
