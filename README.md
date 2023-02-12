# Artificial Particle Life in CUDA

This is a CUDA implementation of [this project](https://github.com/hunar4321/particle-life) by [Hunar Ahmad](https://github.com/hunar4321).
From the orginal repository:
> A simple program to simulate primitive Artificial Life using simple rules of attraction or repulsion among atom-like particles, producing complex self-organzing life-like patterns

## Requirements

This projet requires having a CUDA capable GPU with at least Compute Capability 2.0, the CUDA Toolkit installed and glut.

## Compilation and Execution

    git clone https://github.com/etabeta1/ArtificialParticleLife_CUDA.git
    cd ArtificialParticleLife_CUDA
    make
    make run

If you want to print the time took by your GPU to compute every single frame, just uncomment the ```#define PRINT_TIME``` line at hte top of the ```main.cu``` file.

The simulation parameters are hardcoded, you can find and modify them in the ```getInteractionConstant``` function.
The parameters present right now are from a random simulation.


