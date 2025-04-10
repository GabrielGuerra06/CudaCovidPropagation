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

typedef struct {
    int infected;
    int recovered;
    int fatalities;
} ThreadStats;


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

void initialize_Agents(Agent* agents) {
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
    for (int i = 0; i < N_AGENTS / 100; i++) {

}

__host__ __device__ float distance(Agent a, Agent b) {
    float dist_x = a.x - b.x;
    float dist_y = a.y - b.y;
    return sqrtf(dist_x * dist_x + dist_y * dist_y);
}

__host__ __device__ void  contagious(Agent* agents, int idx, int day, unsigned int seed) {

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
        if (rand_number(0.0f, 1.0f, seed + 1) <= agents[idx].short_dist_prob) {
            agents[idx].x += rand_number(-mov_radius, mov_radius, seed + 2);
            agents[idx].y += rand_number(-mov_radius, mov_radius, seed + 3);
        }
        else {
            agents[idx].x = rand_number(0.0f, sim_area_x, seed + 4);
            agents[idx].y = rand_number(0.0f, sim_area_y, seed + 5);
        }

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
            agents[idx].days_quarantined = 0;
        }
    }
}

__host__ __device__ void  fatalCases(Agent* agents, int idx, int day, unsigned int seed) {
    if (agents[idx].status == IN_QUARANTINE &&
        rand_number(0.0f, 1.0f, seed) <= agents[idx].mort_prob) {
        agents[idx].status = DECEASED;
    }
}

void update_status(Agent* agents, int day) {
    int infected_count = 0;
    int recovered_count = 0;
    int fatalities_count = 0;

    for (int i = 0; i < N_AGENTS; i++) {
        if (agents[i].status == INFECTED || agents[i].status == IN_QUARANTINE) {
            infected_count++;
        }
        else if (agents[i].status == NOT_INFECTED && agents[i].days_quarantined > 0) {
            recovered_count++;
        }
        else if (agents[i].status == DECEASED) {
            fatalities_count++;
        }
    }

    printf("Day %d: %d infected, %d in quarantine, %d deceased\n",
        day, infected_count - fatalities_count,
        infected_count, fatalities_count);

    daily_infected[day] = infected_count - total_infected;
    daily_recovered[day] = recovered_count - total_recovered;
    daily_fatalities[day] = fatalities_count - total_fatalities;

    total_infected = infected_count;
    total_recovered = recovered_count;
    total_fatalities = fatalities_count;

    if (total_infected >= 1 && first_infection_day == -1) {
        first_infection_day = day;
    }
    if (total_infected >= N_AGENTS / 2 && half_population_day == -1) {
        half_population_day = day;
    }
    if (total_infected >= N_AGENTS && full_population_day == -1) {
        full_population_day = day;
    }

    if (total_recovered >= 1 && first_recovery_day == -1) {
        first_recovery_day = day;
    }
    if (total_recovered >= N_AGENTS / 2 && half_recovery_day == -1) {
        half_recovery_day = day;
    }
    if (total_recovered >= N_AGENTS && full_recovery_day == -1) {
        full_recovery_day = day;
    }

    if (total_fatalities >= 1 && first_fatality_day == -1) {
        first_fatality_day = day;
    }
    if (total_fatalities >= N_AGENTS / 2 && half_fatalities_day == -1) {
        half_fatalities_day = day;
    }
    if (total_fatalities >= N_AGENTS && full_fatalities_day == -1) {
        full_fatalities_day = day;
    }
}
void simulate_cpu(Agent* agents) {
    clock_t start = clock();

    for (int day = 0; day < day_max; day++) {
        for (int movement = 0; movement < movs_per_day; movement++) {
            for (int i = 0; i < N_AGENTS; i++) {
                unsigned int seed = time(NULL) + day + movement + i;
                contagious(agents, i, day, seed);
                apply_movility(agents, i, day, seed + 1);
            }
        }

        for (int i = 0; i < N_AGENTS; i++) {
            unsigned int seed = time(NULL) + day + i;
            externalInfection(agents, i, day, seed);
            incubation_rule(agents, i, day, seed + 1);
            fatalCases(agents, i, day, seed + 2);
        }

        update_status(agents, day);
    }

    clock_t end = clock();
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("CPU Simulation Time: %.4f seconds\n", cpu_time);
}

__global__ void simulate_gpu_kernel(Agent* agents, ThreadStats* thread_stats, int day, int movement, int max_movements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_AGENTS) return;

    unsigned int seed = day * 1000 + movement * 100 + idx;

    if (movement < max_movements) {
        contagious(agents, idx, day, seed);
        apply_movility(agents, idx, day, seed + 1);
    }
    else if (movement == max_movements) {
        externalInfection(agents, idx, day, seed + 2);
        incubation_rule(agents, idx, day, seed + 3);
        fatalCases(agents, idx, day, seed + 4);

        thread_stats[idx].infected = (agents[idx].status == INFECTED || agents[idx].status == IN_QUARANTINE);
        thread_stats[idx].recovered = (agents[idx].status == NOT_INFECTED && agents[idx].days_quarantined > 0);
        thread_stats[idx].fatalities = (agents[idx].status == DECEASED);

    }
}

void simulate_gpu(Agent* agents) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    Agent* d_agents;
    cudaMalloc(&d_agents, N_AGENTS * sizeof(Agent));
    cudaMemcpy(d_agents, agents, N_AGENTS * sizeof(Agent), cudaMemcpyHostToDevice);

    ThreadStats* d_thread_stats;
    cudaMalloc(&d_thread_stats, N_AGENTS * sizeof(ThreadStats));



    int blockSize = 256;
    int gridSize = (N_AGENTS + blockSize - 1) / blockSize;
    int* d_block_stats;
    cudaMalloc(&d_block_stats, gridSize * 3 * sizeof(int));

    int* h_block_stats = (int*)malloc(gridSize * 3 * sizeof(int));
    int* h_day_stats = (int*)malloc(3 * sizeof(int));

    for (int day = 0; day < day_max; day++) {
        for (int movement = 0; movement <= movs_per_day; movement++) {
            if (movement == 0) {
                cudaMemset(d_thread_stats, 0, N_AGENTS * sizeof(ThreadStats));
            }

            simulate_gpu_kernel << <gridSize, blockSize >> > (d_agents, d_thread_stats, day, movement, movs_per_day);
            cudaDeviceSynchronize();
        }

        ThreadStats* h_thread_stats = (ThreadStats*)malloc(N_AGENTS * sizeof(ThreadStats));
        cudaMemcpy(h_thread_stats, d_thread_stats, N_AGENTS * sizeof(ThreadStats), cudaMemcpyDeviceToHost);

        int infected = 0, recovered = 0, fatalities = 0;
        for (int i = 0; i < N_AGENTS; i++) {
            infected += h_thread_stats[i].infected;
            recovered += h_thread_stats[i].recovered;
            fatalities += h_thread_stats[i].fatalities;
        }

        daily_infected[day] = infected - total_infected;
        daily_recovered[day] = recovered - total_recovered;
        daily_fatalities[day] = fatalities - total_fatalities;

        total_infected = infected;
        total_recovered = recovered;
        total_fatalities = fatalities;


        int* h_stats = (int*)malloc(day_max * 3 * sizeof(int));
        cudaMemcpy(h_stats, d_thread_stats, day_max * 3 * sizeof(int), cudaMemcpyDeviceToHost);

        for (int day = 0; day < day_max; day++) {
            daily_infected[day] = h_stats[day * 3 + 0] - (day > 0 ? h_stats[(day - 1) * 3 + 0] : 0);
            daily_recovered[day] = h_stats[day * 3 + 1] - (day > 0 ? h_stats[(day - 1) * 3 + 1] : 0);
            daily_fatalities[day] = h_stats[day * 3 + 2] - (day > 0 ? h_stats[(day - 1) * 3 + 2] : 0);

            total_infected = h_stats[day * 3 + 0];
            total_recovered = h_stats[day * 3 + 1];
            total_fatalities = h_stats[day * 3 + 2];

            if (total_infected >= 1 && first_infection_day == -1) first_infection_day = day;
            if (total_infected >= N_AGENTS / 2 && half_population_day == -1) half_population_day = day;
            if (total_infected >= N_AGENTS && full_population_day == -1) full_population_day = day;
            if (total_recovered >= 1 && first_recovery_day == -1) first_recovery_day = day;
            if (total_recovered >= N_AGENTS / 2 && half_recovery_day == -1) half_recovery_day = day;
            if (total_recovered >= N_AGENTS && full_recovery_day == -1) full_recovery_day = day;
            if (total_fatalities >= 1 && first_fatality_day == -1) first_fatality_day = day;
            if (total_fatalities >= N_AGENTS / 2 && half_fatalities_day == -1) half_fatalities_day = day;
            if (total_fatalities >= N_AGENTS && full_fatalities_day == -1) full_fatalities_day = day;
        }

        free(h_stats);
    }
    cudaFree(d_thread_stats);
    cudaFree(d_agents);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU Simulation Time: %.4f seconds\n", gpu_time / 1000.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}




void print_results() {
    printf("\n----------------- Simulation Results -----------------n");
    printf("Total Agents: %d\n", N_AGENTS);
    printf("Simulation Days: %d\n", day_max);
    printf("Movements per Day: %d\n", movs_per_day);

    printf("\nDaily Infection Counts:\n");
    for (int i = 0; i < day_max; i++) {
        printf("Day %2d: %3d new infections, %3d new recoveries, %3d new fatalities\n",
            i + 1, daily_infected[i], daily_recovered[i], daily_fatalities[i]);
    }

    printf("\nCumulative Totals:\n");
    printf("Total Infected: %d\n", total_infected);
    printf("Total Recovered: %d\n", total_recovered);
    printf("Total Fatalities: %d\n", total_fatalities);

    printf("\nInfection Rate:\n");
    printf("First infection on day %d\n", first_infection_day + 1);
    printf("50% of population infected by day %d\n", half_population_day + 1);
    printf("100% of population infected by day %d\n", full_population_day + 1);

    printf("\nRecovery Rate:\n");
    printf("First recovery on day %d\n", first_recovery_day + 1);
    printf("50%% of population recovered by day %d\n", half_recovery_day + 1);
    printf("100%% of population recovered by day %d\n", full_recovery_day + 1);

    printf("\nFatality Rate:\n");
    printf("First fatality on day %d\n", first_fatality_day + 1);
    printf("50%% of fatalities occurred by day %d\n", half_fatalities_day + 1);
    printf("100%% of fatalities occurred by day %d\n", full_fatalities_day + 1);
}


int main()
{
    Agent* agents = (Agent*)malloc(N_AGENTS * sizeof(Agent));
    initialize_Agents(agents);

    printf("CPU simulation...\n");
    simulate_cpu(agents);

    memset(daily_infected, 0, sizeof(daily_infected));
    memset(daily_recovered, 0, sizeof(daily_recovered));
    memset(daily_fatalities, 0, sizeof(daily_fatalities));
    total_infected = 0;
    total_recovered = 0;
    total_fatalities = 0;
    first_infection_day = -1;
    half_population_day = -1;
    full_population_day = -1;
    first_recovery_day = -1;
    half_recovery_day = -1;
    full_recovery_day = -1;
    first_fatality_day = -1;
    half_fatalities_day = -1;
    full_fatalities_day = -1;

    printf("GPU simulation...\n");
    simulate_gpu(agents);

    print_results();

    free(agents);
    return 0;
}
