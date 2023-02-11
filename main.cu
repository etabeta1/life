// TODO: Use streams to update and draw in parallel

#define GL_GLEXT_PROTOTYPES
#define PRINT_TIME

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
uchar4 *d_screen;

Particle *d_particles, *d_new_particles;
Particle *d_group_1, *d_group_2, *d_group_3, *d_group_4;

#ifdef PRINT_TIME
cudaEvent_t initialize_start, initialize_stop;
cudaEvent_t update_start, update_stop;
cudaEvent_t draw_start, draw_stop;
#endif

__global__ void initializeParticles(Particle* particles, int kind, int n);
__global__ void update_particles_speed(Particle* particles, Particle* new_particles, int n);
__global__ void update_particles_position(Particle* particles, int n);
__global__ void draw_particles(Particle* particles, int n, uchar4* screen);

static void display_func();
static void key_func(unsigned char key, int x, int y);
static void idle_func();

void chooseDevice(int major, int minor);
#ifdef PRINT_TIME
void printTimeBetweenEvents(cudaEvent_t start, cudaEvent_t stop, const char* message);
#endif

int main(int argc, char** argv) {
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

    cudaFree(0);

    #ifdef PRINT_TIME
    // Create events for timing
    cudaEventCreate(&initialize_start);
    cudaEventCreate(&initialize_stop);
    cudaEventCreate(&update_start);
    cudaEventCreate(&update_stop);
    cudaEventCreate(&draw_start);
    cudaEventCreate(&draw_stop);
    #endif

    // Initialize all the particles and divide them into groups
    #ifdef PRINT_TIME
    cudaEventRecord(initialize_start);
    #endif

    cudaMalloc((void**)&d_particles, PARTICLE_COUNT * sizeof(Particle));
    d_group_1 = d_particles;
    d_group_2 = d_particles + PARTICLE_COUNT / 4;
    d_group_3 = d_particles + PARTICLE_COUNT / 2;
    d_group_4 = d_particles + PARTICLE_COUNT * 3 / 4;
    cudaMalloc((void**)&d_new_particles, PARTICLE_COUNT * sizeof(Particle));

    initializeParticles<<<PARTICLE_COUNT / 256, 256>>>(d_group_1, 1, PARTICLE_COUNT / 4);
    initializeParticles<<<PARTICLE_COUNT / 256, 256>>>(d_group_2, 2, PARTICLE_COUNT / 4);
    initializeParticles<<<PARTICLE_COUNT / 256, 256>>>(d_group_3, 3, PARTICLE_COUNT / 4);
    initializeParticles<<<PARTICLE_COUNT / 256, 256>>>(d_group_4, 4, PARTICLE_COUNT / 4);

    #ifdef PRINT_TIME
    cudaEventRecord(initialize_stop);
    cudaEventSynchronize(initialize_stop);

    printTimeBetweenEvents(initialize_start, initialize_stop, "Particle initialization time:");
    #endif

    // END OF CUDA THINGS

    cudaGraphicsUnmapResources(1, &resource, NULL);

    // glut callbacks for display, keyboard and update of the particles
    glutKeyboardFunc(key_func);
    glutDisplayFunc(display_func);
    glutIdleFunc(idle_func);

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

__global__ void update_particles_speed(Particle* particles, Particle* new_particles, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // record interactions from particles, store the results in new_particles

}

__global__ void update_particles_position(Particle* particles, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n) {
        particles[i].x += particles[i].vx;
        particles[i].y += particles[i].vy;

        if(particles[i].x < 0) {
            particles[i].x = 0;
        }
        if(particles[i].x >= WINDOW_SIZE) {
            particles[i].x = WINDOW_SIZE - 1;
        }
        if(particles[i].y < 0) {
            particles[i].y = 0;
        }
        if(particles[i].y >= WINDOW_SIZE) {
            particles[i].y = WINDOW_SIZE - 1;
        }
    }
}

__global__ void draw_particles(Particle* particles, int n, uchar4* screen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n) {
        int x = particles[i].x;
        int y = particles[i].y;

        if(x >= 0 && x < WINDOW_SIZE && y >= 0 && y < WINDOW_SIZE) {
            int index = y * WINDOW_SIZE + x;

            switch(particles[i].kind) {
                case 1:
                    screen[index].x = 255;
                    screen[index].y = 0;
                    screen[index].z = 0;
                    screen[index].w = 255;
                    break;
                case 2:
                    screen[index].x = 0;
                    screen[index].y = 255;
                    screen[index].z = 0;
                    screen[index].w = 255;
                    break;
                case 3:
                    screen[index].x = 0;
                    screen[index].y = 0;
                    screen[index].z = 255;
                    screen[index].w = 255;
                    break;
                case 4:
                    screen[index].x = 255;
                    screen[index].y = 255;
                    screen[index].z = 255;
                    screen[index].w = 255;
                    break;
            }
        }
    }
}

static void display_func() {
    glClearColor(.0, .0, .0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);                           // Clear the screen

    glDrawPixels(WINDOW_SIZE, WINDOW_SIZE, GL_RGBA, GL_UNSIGNED_BYTE, 0); // Draw the buffer
    glutSwapBuffers();                                      // Swap the buffers
}

static void key_func(unsigned char key, int x, int y) {
    if(key == 27) {
        cudaGraphicsUnregisterResource(resource);       // Unregister the resource
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);    // Unbind the buffer
        glDeleteBuffers(1, &buffer);                    // Delete the buffer

        cudaFree(d_particles);                          // Free the memory
        cudaFree(d_new_particles);

        #ifdef PRINT_TIME
        cudaEventDestroy(initialize_start);
        cudaEventDestroy(initialize_stop);
        cudaEventDestroy(update_start);
        cudaEventDestroy(update_stop);
        cudaEventDestroy(draw_start);
        cudaEventDestroy(draw_stop);
        #endif

        exit(0);
    }
}

static void idle_func() {
    dim3 update_speed_blocks((PARTICLE_COUNT + 255) / 256, (PARTICLE_COUNT + 255) / 256);
    dim3 update_speed_threads(256, 256);

    dim3 update_position_blocks((PARTICLE_COUNT + 255) / 256);
    dim3 update_position_threads(256);

    #ifdef PRINT_TIME
    cudaEventRecord(update_start);
    #endif

    cudaMemcpy(d_new_particles, d_particles, sizeof(Particle) * PARTICLE_COUNT, cudaMemcpyDeviceToDevice);
    update_particles_speed<<<update_speed_blocks, update_speed_threads>>>(d_particles, d_new_particles, PARTICLE_COUNT);
    update_particles_position<<<update_position_blocks, update_position_threads>>>(d_particles, PARTICLE_COUNT);
    cudaMemcpy(d_particles, d_new_particles, sizeof(Particle) * PARTICLE_COUNT, cudaMemcpyDeviceToDevice);

    #ifdef PRINT_TIME
    cudaEventRecord(update_stop);
    cudaEventSynchronize(update_stop);
    printTimeBetweenEvents(update_start, update_stop, "Update time: ");
    #endif

    dim3 draw_blocks((PARTICLE_COUNT + 255) / 256);
    dim3 draw_threads(256);

    
    cudaGraphicsMapResources(1, &resource, NULL);
    draw_particles<<<draw_blocks, draw_threads>>>(d_particles, PARTICLE_COUNT, d_screen);
    cudaGraphicsUnmapResources(1, &resource, NULL);

    glutPostRedisplay();
}

// Function which chooses the device to use given the compute capability, in my case I have a GT730 with compute capability 3.5
void chooseDevice(int major, int minor) {
    cudaDeviceProp prop;
    int dev;

    memset(&prop, 0, sizeof(cudaDeviceProp));

    prop.major = major;
    prop.minor = minor;

    cudaChooseDevice(&dev, &prop);
    cudaGLSetGLDevice(dev);
}

#ifdef PRINT_TIME
void printTimeBetweenEvents(cudaEvent_t start, cudaEvent_t stop, const char* message) {
    float time;

    cudaEventElapsedTime(&time, start, stop);

    printf("%s: %f ms\n", message, time);
}
#endif
