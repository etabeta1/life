IN=main.cu
OUT=main

all: $(OUT)

$(OUT): $(IN)
	nvcc -o $(OUT) -lcuda -lcurand -lGL -lGLU -lglut -arch=sm_35 -rdc=true $(IN) -Wno-deprecated-gpu-targets -Wno-deprecated-declarations

run: all
	./main

prof: all
	nsys profile ./main

clean:
	rm -f *.o main *.nsys-rep ./temp/* 

.PHONY: run prof clean