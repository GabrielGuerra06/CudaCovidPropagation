// Sequential (CPU) simulation
void simulate_cpu(Agent* agents) {
    clock_t start = clock();
    
    for (int day = 0; day < D_MAX; day++) {
        for (int movement = 0; movement < M_MAX; movement++) {
            for (int i = 0; i < N; i++) {
                unsigned int seed = time(NULL) + day + movement + i;
                apply_contagion_rule(agents, i, day, seed);
                apply_mobility_rule(agents, i, day, seed+1);
            }
        }
        
        for (int i = 0; i < N; i++) {
            unsigned int seed = time(NULL) + day + i;
            apply_external_contagion_rule(agents, i, day, seed);
            apply_incubation_recovery_rule(agents, i, day, seed+1);
            apply_fatality_rule(agents, i, day, seed+2);
        }
        
        update_statistics(agents, day);
    }
    
    clock_t end = clock();
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("CPU Simulation Time: %.4f seconds\n", cpu_time);
}

// Kernel for parallel simulation
_global_ void simulate_gpu_kernel(Agent* agents, int* stats, int day, int movement, int max_movements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    unsigned int seed = day * 1000 + movement * 100 + idx;
    
    // Apply rules based on simulation phase
    if (movement < max_movements) {
        // During the day (movements)
        apply_contagion_rule(agents, idx, day, seed);
        apply_mobility_rule(agents, idx, day, seed+1);
    } else if (movement == max_movements) {
        // End of day (external contagion, incubation, fatality)
        apply_external_contagion_rule(agents, idx, day, seed+2);
        apply_incubation_recovery_rule(agents, idx, day, seed+3);
        apply_fatality_rule(agents, idx, day, seed+4);
        
        // Update statistics (only once per day)
        int infected = (agents[idx].status == INFECTED || agents[idx].status == QUARANTINED);
        int recovered = (agents[idx].status == NOT_INFECTED && agents[idx].days_quarantined > 0);
        int deceased = (agents[idx].status == DECEASED);
        
        atomicAdd(&stats[day*3 + 0], infected);          // Daily infected
        atomicAdd(&stats[day*3 + 1], recovered);         // Daily recovered
        atomicAdd(&stats[day*3 + 2], deceased);          // Daily fatalities
    }
}

// Parallel (GPU) simulation
void simulate_gpu(Agent* agents) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Allocate device memory
    Agent* d_agents;
    cudaMalloc(&d_agents, N * sizeof(Agent));
    cudaMemcpy(d_agents, agents, N * sizeof(Agent), cudaMemcpyHostToDevice);
    
    // Allocate device memory for statistics
    int* d_stats;
    cudaMalloc(&d_stats, D_MAX * 3 * sizeof(int));
    cudaMemset(d_stats, 0, D_MAX * 3 * sizeof(int));
    
    // Set up grid and block dimensions
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Run simulation
    for (int day = 0; day < D_MAX; day++) {
        for (int movement = 0; movement <= M_MAX; movement++) {
            simulate_gpu_kernel<<<gridSize, blockSize>>>(d_agents, d_stats, day, movement, M_MAX);
            cudaDeviceSynchronize();
        }
    }
    
    // Copy statistics back to host
    int* h_stats = (int*)malloc(D_MAX * 3 * sizeof(int));
    cudaMemcpy(h_stats, d_stats, D_MAX * 3 * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Process statistics
    for (int day = 0; day < D_MAX; day++) {
        daily_infected[day] = h_stats[day*3 + 0] - (day > 0 ? h_stats[(day-1)*3 + 0] : 0);
        daily_recovered[day] = h_stats[day*3 + 1] - (day > 0 ? h_stats[(day-1)*3 + 1] : 0);
        daily_fatalities[day] = h_stats[day*3 + 2] - (day > 0 ? h_stats[(day-1)*3 + 2] : 0);
        
        total_infected = h_stats[day*3 + 0];
        total_recovered = h_stats[day*3 + 1];
        total_fatalities = h_stats[day*3 + 2];
        
        // Record milestone days (same as CPU version)
        if (total_infected >= 1 && first_infection_day == -1) first_infection_day = day;
        if (total_infected >= N/2 && half_population_day == -1) half_population_day = day;
        if (total_infected >= N && full_population_day == -1) full_population_day = day;
        if (total_recovered >= 1 && first_recovery_day == -1) first_recovery_day = day;
        if (total_recovered >= N/2 && half_recovery_day == -1) half_recovery_day = day;
        if (total_recovered >= N && full_recovery_day == -1) full_recovery_day = day;
        if (total_fatalities >= 1 && first_fatality_day == -1) first_fatality_day = day;
        if (total_fatalities >= N/2 && half_fatalities_day == -1) half_fatalities_day = day;
        if (total_fatalities >= N && full_fatalities_day == -1) full_fatalities_day = day;
    }
    
    free(h_stats);
    cudaFree(d_stats);
    cudaFree(d_agents);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU Simulation Time: %.4f seconds\n", gpu_time / 1000.0f);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
