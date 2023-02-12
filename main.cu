//#define PRINT_TIME // Uncomment this line to print the time of each step

#include <stdio.h>

#define GL_GLEXT_PROTOTYPES
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include <GL/glut.h>

#define WINDOW_SIZE 512
#define PARTICLE_COUNT 256              // Crashes if less than 4
#define MAX_INTERACTION_DISTANCE 64
#define BORDER_DISTANCE 64              // Distance from the borders where the particles wont be created at the beginning

enum ParticleKind { KIND_RED, KIND_GREEN, KIND_BLUE, KIND_YELLOW };

struct Particle {
    float x, y;         // Position
    float vx, vy;       // Speed
    ParticleKind kind;  // Kind of the particle
};

GLuint buffer;                  // OpenGL buffer
cudaGraphicsResource *resource; // CUDA resource
uchar4 *d_screen;               // Pointer to the screen buffer in the GPU

Particle *d_particles;
Particle *d_group_1, *d_group_2, *d_group_3, *d_group_4;

#ifdef PRINT_TIME
cudaEvent_t initialize_start, initialize_stop;
cudaEvent_t update_start, update_stop;
cudaEvent_t draw_start, draw_stop;
#endif

// CUDA kernels
__global__ void initializeParticles(Particle* particles, ParticleKind kind, int n);
__global__ void update_particles_speed(Particle* particles, int n);
__global__ void update_particles_position(Particle* particles, int n);
__global__ void draw_particles(Particle* particles, int n, uchar4* screen);

// glut callbacks
static void display_func();
static void key_func(unsigned char key, int x, int y);
static void idle_func();

// Other functions
void chooseDevice(int major, int minor);
#ifdef PRINT_TIME
void printTimeBetweenEvents(cudaEvent_t start, cudaEvent_t stop, const char* message);
#endif

int main(int argc, char** argv) {
    size_t size;
    cudaError_t err;
    
    chooseDevice(3, 5);     // Choose a device with compute capability 3.5 (if program crashes, change the version according to your device but check to have at least 2.0)

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

    initializeParticles<<<(PARTICLE_COUNT + 255) / 256, 256>>>(d_group_1, KIND_RED, PARTICLE_COUNT / 4);
    initializeParticles<<<(PARTICLE_COUNT + 255) / 256, 256>>>(d_group_2, KIND_GREEN, PARTICLE_COUNT / 4);
    initializeParticles<<<(PARTICLE_COUNT + 255) / 256, 256>>>(d_group_3, KIND_BLUE, PARTICLE_COUNT / 4);
    initializeParticles<<<(PARTICLE_COUNT + 255) / 256, 256>>>(d_group_4, KIND_YELLOW, PARTICLE_COUNT / 4);

    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Couldn't initialize particles: %s\n", cudaGetErrorString(err));
        exit(1);
    }

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

__global__ void initializeParticles(Particle* particles, ParticleKind kind, int n) {
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

__device__ float getInteractionConstant(ParticleKind kind1, ParticleKind kind2) {
    float matrix[4][4] = {
        { 0.926139214076102, -0.834165324456992,  0.280928927473724, -0.064273079857230},
        {-0.461709646508098,  0.491424346342683,  0.276072602719069,  0.641348738688976},
        { 0.280928927473724,  0.276072602719069,  0.491424346342683, -0.461709646508098},
        {-0.064273079857230,  0.641348738688976, -0.461709646508098,  0.926139214076102}
    };
    return matrix[kind1][kind2];
}

__global__ void update_particles_speed(Particle* particles, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < n && y < n) {
        float fx = .0, fy = .0;
        float dx = particles[x].x - particles[y].x;
        float dy = particles[x].y - particles[y].y;
        float d = sqrtf(dx * dx + dy * dy);
        if(d > 0 && d < MAX_INTERACTION_DISTANCE) {
            float F = getInteractionConstant(particles[x].kind, particles[y].kind) / d;
            fx += F * dx;
            fy += F * dy;
        }

        atomicAdd(&particles[x].vx, fx);
        atomicAdd(&particles[x].vy, fy);
    }
}

__global__ void update_particles_position(Particle* particles, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n) {
        particles[i].vx /= 2.0;
        particles[i].vy /= 2.0;
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
            int position = y * WINDOW_SIZE + x;

            switch(particles[i].kind) {
                case KIND_RED:
                    screen[position].x = 255;
                    screen[position].y = 0;
                    screen[position].z = 0;
                    screen[position].w = 255;
                    break;
                case KIND_GREEN:
                    screen[position].x = 0;
                    screen[position].y = 255;
                    screen[position].z = 0;
                    screen[position].w = 255;
                    break;
                case KIND_BLUE:
                    screen[position].x = 0;
                    screen[position].y = 0;
                    screen[position].z = 255;
                    screen[position].w = 255;
                    break;
                case KIND_YELLOW:
                    screen[position].x = 255;
                    screen[position].y = 255;
                    screen[position].z = 0;
                    screen[position].w = 255;
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
    cudaError_t err;

    dim3 update_speed_blocks((PARTICLE_COUNT + 31) / 32, (PARTICLE_COUNT + 31) / 32);
    dim3 update_speed_threads(32, 32);

    dim3 update_position_blocks((PARTICLE_COUNT + 255) / 256);
    dim3 update_position_threads(256);

    #ifdef PRINT_TIME
    cudaEventRecord(update_start);
    #endif

    update_particles_speed<<<update_speed_blocks, update_speed_threads>>>(d_particles, PARTICLE_COUNT);
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Couldn't update the particles\' speed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    update_particles_position<<<update_position_blocks, update_position_threads>>>(d_particles, PARTICLE_COUNT);
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Couldn\'t update the particles\' position: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    #ifdef PRINT_TIME
    cudaEventRecord(update_stop);
    cudaEventSynchronize(update_stop);
    printTimeBetweenEvents(update_start, update_stop, "Update time");
    #endif

    dim3 draw_blocks((PARTICLE_COUNT + 255) / 256);
    dim3 draw_threads(256);
    
    cudaGraphicsMapResources(1, &resource, NULL);
    cudaMemset(d_screen, 0, WINDOW_SIZE * WINDOW_SIZE * sizeof(uchar4));
    draw_particles<<<draw_blocks, draw_threads>>>(d_particles, PARTICLE_COUNT, d_screen);
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Couldn\'t draw the particles: %s\n", cudaGetErrorString(err));
        exit(1);
    }

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
