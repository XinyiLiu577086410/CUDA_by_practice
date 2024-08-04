#Makefile


SOURCE =  01histogram_gmem.cu
FLAGS = -arch=sm_89
CC = nvcc  

exe: $(SOURCE)
	$(CC) $(FLAGS) $(SOURCE)  -o exe 


clean:
	$(RM) -rf exe 

run:
	./exe




