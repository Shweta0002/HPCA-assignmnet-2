sim: data/3.in data/5.in data/7.in data/9.in data/11.in data/13.in data/64.in data/128.in data/1024.in data/2048.in data/4096.in data/8192.in data/16384.in dilated_conv_sim

server: data/1024.in dilated_conv_server

dilated_conv_server: main.cu header/gpu_thread.h
	nvcc main.cu -o dilated_conv_server -I ./header

dilated_conv_sim: main.cu header/gpu_thread.h
	nvcc main.cu -o dilated_conv -I ./header --cudart shared -gencode arch=compute_70,code=compute_70

data/generate: data/generate.cpp
	g++ ./data/generate.cpp -o ./data/generate

data/3.in: data/generate
	./data/generate 3

data/5.in: data/generate
	./data/generate 5

data/7.in: data/generate
	./data/generate 7

data/9.in: data/generate
	./data/generate 9

data/11.in: data/generate
	./data/generate 11

data/13.in: data/generate
	./data/generate 13

data/64.in: data/generate
	./data/generate 64

data/128.in: data/generate
	./data/generate 128

data/1024.in: data/generate
	./data/generate 1024

data/2048.in: data/generate
	./data/generate 2048

data/4096.in: data/generate
	./data/generate 4096 

data/8192.in: data/generate
	./data/generate 8192 

data/16384.in: data/generate
	./data/generate 16384 


run_sim: data/3.in data/5.in data/7.in data/9.in data/11.in data/13.in data/64.in data/128.in data/1024.in data/2048.in data/4096.in data/8192.in data/16384.in dilated_conv_sim

	
	./dilated_conv_sim -i data/16384.in -k data/9.in
	./dilated_conv_sim -i data/16384.in -k data/11.in
	./dilated_conv_sim -i data/16384.in -k data/13.in
	
	
run_server: data/128.in dilated_conv_server
	
	./dilated_conv_server -i data/16384.in -k data/11.in
	./dilated_conv_server -i data/16384.in -k data/13.in
	

clean:
	rm -f dilated_conv_sim dilated_conv_server
