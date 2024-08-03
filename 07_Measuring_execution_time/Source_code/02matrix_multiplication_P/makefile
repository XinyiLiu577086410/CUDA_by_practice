#Makefile


SOURCE =  02matrix_multiplication.cu

CC = nvc++

exe: $(SOURCE)
	$(CC) $(SOURCE)  -o exe -g -O0 -fopenmp -cuda

clean:
	$(RM) -rf exe 

run:
	./exe




