// TODO: Use streams to update and draw in parallel

#define GL_GLEXT_PROTOTYPES

#include <stdio.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include <GL/glut.h>

#define WINDOW_SIZE 512
#define PARTICLE_COUNT 256

#define BORDER_DISTANCE 64

struct Particle {
    float x, y;
    float vx, vy;
    int kind;
};

GLuint buffer;
cudaGraphicsResource *resource;

__global__ void initializeParticles(Particle* particles, int kind, int n);
__global__ void update_particles(Particle* particles, int n);
__global__ void draw_particles(Particle* particles, int n, uchar4* screen);

static void display_func();
static void key_func(unsigned char key, int x, int y);

void chooseDevice(int major, int minor);

int main(int argc, char** argv) {
    Particle *d_particles;
    Particle *d_group_1, *d_group_2, *d_group_3, *d_group_4;
    uchar4 screen, *d_screen;
    size_t size;
    
    chooseDevice(3, 5);

    // Initialize GLUT and create window
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA); // Double buffering and RGBA
    glutInitWindowSize(WINDOW_SIZE, WINDOW_SIZE);
    glutCreateWindow("Artificial particle life");

    // Initialize buffer
    glGenBuffers(1, &buffer);                                   // Generate a buffer
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, buffer);           // Bind it to the buffer variable
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, WINDOW_SIZE * WINDOW_SIZE * sizeof(int), NULL, GL_DYNAMIC_DRAW_ARB); // Allocate the memory for the buffer

    // After these three lines, every time we write to the d_screen pointer, the buffer will be updated
    cudaGraphicsGLRegisterBuffer(&resource, buffer, cudaGraphicsMapFlagsNone);
    cudaGraphicsMapResources(1, &resource, NULL);
    cudaGraphicsResourceGetMappedPointer((void**)&d_screen, &size, resource);

    // CUDA THINGS HERE

    // Initialize all the particles and divide them into groups
    cudaMalloc((void**)&d_particles, PARTICLE_COUNT * sizeof(Particle));
    d_group_1 = d_particles;
    d_group_2 = d_particles + PARTICLE_COUNT / 4;
    d_group_3 = d_particles + PARTICLE_COUNT / 2;
    d_group_4 = d_particles + PARTICLE_COUNT * 3 / 4;

    initializeParticles<<<PARTICLE_COUNT / 256, 256>>>(d_group_1, 1, PARTICLE_COUNT / 4);
    initializeParticles<<<PARTICLE_COUNT / 256, 256>>>(d_group_2, 2, PARTICLE_COUNT / 4);
    initializeParticles<<<PARTICLE_COUNT / 256, 256>>>(d_group_3, 3, PARTICLE_COUNT / 4);
    initializeParticles<<<PARTICLE_COUNT / 256, 256>>>(d_group_4, 4, PARTICLE_COUNT / 4);

    // END OF CUDA THINGS

    cudaGraphicsUnmapResources(1, &resource, NULL);

    // glut callbacks for display and keyboard
    glutKeyboardFunc(key_func);
    glutDisplayFunc(display_func);

    // Start the main loop
    glutMainLoop();

    return 0;
}

__global__ void initializeParticles(Particle* particles, int kind, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n) {
        curandState state;

        curand_init(clock64(), i, 0, &state);

        particles[i].x = curand_uniform(&state) * (WINDOW_SIZE - BORDER_DISTANCE * 2) + BORDER_DISTANCE;
        particles[i].y = curand_uniform(&state) * (WINDOW_SIZE - BORDER_DISTANCE * 2) + BORDER_DISTANCE;
        particles[i].vx = 0;
        particles[i].vy = 0;
        particles[i].kind = kind;
    }
}

__global__ void update_particles(Particle* particles, int n) {
    //throw "Not implemented";
}

__global__ void draw_particles(Particle* particles, int n, uchar4* screen) {
    //throw "Not implemented";
}

static void display_func() {
    glClear(GL_COLOR_BUFFER_BIT);                           // Clear the screen

    // TODO: Update particles
    // TODO: Draw particles

    glDrawPixels(WINDOW_SIZE, WINDOW_SIZE, GL_RGBA, GL_UNSIGNED_BYTE, 0); // Draw the buffer
    glutSwapBuffers();                                      // Swap the buffers
}

static void key_func(unsigned char key, int x, int y) {
    if(key == 27) {
        cudaGraphicsUnregisterResource(resource);       // Unregister the resource
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);    // Unbind the buffer
        glDeleteBuffers(1, &buffer);                    // Delete the buffer
    }
}

// Function which chooses the device to use given the compute capabilities, in my case I have a GT730 with compute capability 3.5
void chooseDevice(int major, int minor) {
    cudaDeviceProp prop;
    int dev;

    memset(&prop, 0, sizeof(cudaDeviceProp));

    prop.major = major;
    prop.minor = minor;

    cudaChooseDevice(&dev, &prop);
    cudaGLSetGLDevice(dev);
}