#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <chrono>
#include <cuda.h>
#include <curand_kernel.h>
#include <math.h>

using namespace std;
using namespace std::chrono;



#define  N_AGENTS 1024
#define  day_max  30
#define  movs_per_day 10
#define  mov_radius  5.0f //m
#define  contg_dist  1.0f //m
#define  sim_area_x  500
#define  sim_area_y  500



//states for the Agents
#define NOT_INFECTED 0
#define INFECTED 1
#define IN_QUARANTINE -1
#define DECEASED -2

typedef struct {
    float contg_prob;
    float  external_prob_contg;
    float mort_prob;
    float mov_prob;
    float short_dist_prob;
    int incub_day;
    int recov_day;
    int status;
    int days_infected;
    int days_quarantined;
    float x;
    float y;

} Agent;


int daily_infected[day_max] = { 0 };
int daily_recovered[day_max] = { 0 };
int daily_fatalities[day_max] = { 0 };
int total_infected = 0;
int total_recovered = 0;
int total_fatalities = 0;
int first_infection_day = -1;
int half_population_day = -1;
int full_population_day = -1;
int first_recovery_day = -1;
int half_recovery_day = -1;
int full_recovery_day = -1;
int first_fatality_day = -1;
int half_fatalities_day = -1;
int full_fatalities_day = -1;


__host__ __device__ float rand_number(float min, float max, unsigned int seed) {
    //Function used by chatgpt to generate pseudo random float numbers
    unsigned int state = seed * 1664525 + 1013904223;
    return min + (max - min) * ((float)state / (float)UINT_MAX);
}




__host__ void checkCudaError(const char* error_message) {
    cudaError_t error;
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        cerr << error_message << " - " << cudaGetErrorString(error) << endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void initialize_Agents(Agent* agents) {
    for (int i = 0; i < N_AGENTS; i++) {
        unsigned int seed = time(NULL) + i;

        agents[i].contg_prob = rand_number(0.02f, 0.03f, seed++);
        agents[i].external_prob_contg = rand_number(0.02f, 0.03f, seed++);
        agents[i].mort_prob = rand_number(0.007f, 0.07f, seed++);
        agents[i].mov_prob = rand_number(0.3f, 0.5f, seed++);
        agents[i].short_dist_prob = rand_number(0.7f, 0.9f, seed++);
        agents[i].incub_day = (int)rand_number(5.0f, 6.0f, seed++);
        agents[i].recov_day = 14;
        agents[i].status = NOT_INFECTED;
        agents[i].days_infected = 0;
        agents[i].days_quarantined = 0;
        agents[i].x = rand_number(0.0f, sim_area_x, seed++);
        agents[i].y = rand_number(0.0f, sim_area_y, seed++);
    }
    // agents[0].status = INFECTED;

}

__host__ __device__ float distance(Agent a, Agent b) {
    float dist_x = a.x - b.x;
    float dist_y = a.y - b.y;
    return sqrtf(dist_x * dist_x + dist_y * dist_y);
}

__host__ __device__ float  contagious(Agent* agents, int idx, int day, unsigned int seed) {

    if (agents[idx].status != NOT_INFECTED) return;

    int near_infected = 0;
    for (int i = 0; i < N_AGENTS; i++) {
        if (i == idx) {
            continue;
        }
        if (agents[i].status > 0 && distance(agents[idx], agents[i]) <= contg_dist) {
            near_infected = 1;
            break;
        }
    }

    if (near_infected && rand_number(0.0f, 1.0f, seed) <= agents[idx].contg_prob) {
        agents[idx].status = INFECTED;
        agents[idx].days_infected = 0;

    }
}

__host__ __device__ void apply_movility(Agent* agents, int idx, int day, unsigned int seed) {
    if (agents[idx].status == DECEASED || agents[idx].status == IN_QUARANTINE) return;
    if (rand_number(0.0f, 1.0f, seed) <= agents[idx].mov_prob) {
        if (rand_number(0.0f, 1.0f, seed + 1) <= agents[idx].mov_prob) {
            // Local movement
            agents[idx].x += rand_number(-mov_radius, mov_radius, seed + 2);
            agents[idx].y += rand_number(-mov_radius, mov_radius, seed + 3);
        }
        else {
            // Long distance movement
            agents[idx].x = rand_number(0.0f, sim_area_x, seed + 4);
            agents[idx].y = rand_number(0.0f, sim_area_y, seed + 5);
        }

        // Ensure agents stay within bounds
        agents[idx].x = fmaxf(0.0f, fminf(sim_area_x, agents[idx].x));
        agents[idx].y = fmaxf(0.0f, fminf(sim_area_y, agents[idx].y));
    }

}

__host__ __device__ void externalInfection(Agent* agents, int idx, int day, unsigned int seed) {
    if (agents[idx].status == NOT_INFECTED && rand_number(0.0f, 1.0f, seed) <= agents[idx].external_prob_contg) {
        agents[idx].status = INFECTED;
        agents[idx].days_infected = 0;
    }
}

__host__ __device__ void incubation_rule(Agent* agents, int idx, int day, unsigned int seed)
{
    if (agents[idx].status == INFECTED) {
        agents[idx].days_infected++;

        if (agents[idx].days_infected >= agents[idx].incub_day) {
            agents[idx].status = IN_QUARANTINE;
            agents[idx].days_quarantined = 0;
        }
    }
    else if (agents[idx].status == IN_QUARANTINE) {
        agents[idx].days_quarantined++;

        if (agents[idx].days_quarantined >= agents[idx].recov_day) {
            agents[idx].status = NOT_INFECTED;
        }
    }
}

__global__ void fatalCases(Agent* agents, int idx, int day, unsigned int seed) {
    if (agents[idx].status == IN_QUARANTINE)
}



int main()
{
    return 0;
}
