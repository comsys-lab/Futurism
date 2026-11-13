/*
 * Modified by: Jaehyeok-Ryu (Futurism).
 * Based on comsys-lab/Kubism CPU functions; modified in Futurism.
 * Futurism changes include sample-wise access, ARM NEON distance routines
 * and batch4 processing. The original implementation credit is retained below.
 * Pinned sources and file comparisons: docs/SOURCE_AUDIT.md.
 * Source attribution and license scope: docs/PROVENANCE.md.
 */

/********************************************
*
*Kubism - CPU Function
*Implemented by: Seondeok Kim
*
********************************************/

#define SAMPLE_WISE 1
#define USE_NEON 1 // Set to 1 to use Neon SIMD, 0 for scalar distance_t_squared

#include "kmeans_cpu.h"
#include <omp.h>
#include <vector>
#include <thread>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>


template <typename F>
inline float distance_t_squared(
    const float *v1,
    const float *v2,
    uint32_t v1_size,
    uint16_t features_size,
    uint32_t v1_index) {
    
    float dist = 0.0f;
    for (uint16_t f = 0; f < features_size; f++) {
#if SAMPLE_WISE
        float d = v1[v1_index * features_size + f] - v2[f];
#else
        float d = v1[v1_size * f + v1_index] - v2[f];
#endif
        dist += d * d;
    }
    return dist;
}

template <typename F>
inline float distance_t_neon_squared(
    const float *samples,   // v1
    const float *centroids, // v2
    uint32_t samples_size,  // v1_size
    uint16_t features_size, // features_size
    uint32_t sample         // v1_index
    ) {
    // SIMD accumulation vector 
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    
    uint16_t f = 0;
#pragma GCC unroll 4
    for (; f <= (int)features_size - 4; f += 4) {
#if SAMPLE_WISE
        // neon_sample-wise
        float32x4_t vec_sample = vld1q_f32(samples + sample * features_size + f);
#else
        // neon
        float32x4_t vec_sample = {
            samples[samples_size * (f + 0) + sample],
            samples[samples_size * (f + 1) + sample],
            samples[samples_size * (f + 2) + sample],
            samples[samples_size * (f + 3) + sample]
        };
#endif
        float32x4_t vec_centroid = vld1q_f32(centroids + f);
        float32x4_t diff = vsubq_f32(vec_sample, vec_centroid);
        sum_vec = vaddq_f32(sum_vec, vmulq_f32(diff, diff));
    }
    
    // Horizontally sum the four float values in sum_vec
    float32x2_t sum_pair = vadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
    float sum = vget_lane_f32(sum_pair, 0) + vget_lane_f32(sum_pair, 1);
    
    // Process any remaining features
    for (; f < features_size; f++) {
#if SAMPLE_WISE
        float d = samples[sample * features_size + f] - centroids[f];
#else
        float d = samples[samples_size * f + sample] - centroids[f];
#endif
        sum += d * d;
    }
    
    return sum;
}

template <typename F>
void distance_t_neon_batch4_squared(
    const float *interleaved_samples,
    const float *centroids,
    uint16_t features_size,
    float *dists_sq
    ) {
    
    // Accumulate results for 4 samples in a single SIMD vector
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    
    uint16_t f = 0;
#pragma GCC unroll 4
    for (; f < features_size; f++) {
        // Load f-th feature for 4 interleaved samples at once using SIMD
        // Layout: [F0_S0, F0_S1, F0_S2, F0_S3, F1_S0, F1_S1, ...]
        float32x4_t vec_sample = vld1q_f32(interleaved_samples + f * 4);
        
        // Broadcast centroid feature value to all 4 lanes
        float32x4_t vec_centroid = vdupq_n_f32(centroids[f]);
        
        float32x4_t diff = vsubq_f32(vec_sample, vec_centroid);
        sum_vec = vaddq_f32(sum_vec, vmulq_f32(diff, diff));
    }
    
    // Write out the 4 squared distances
    vst1q_f32(dists_sq, sum_vec);
}


template <typename F>
void kubism_CPU_WB_HETD(
    const uint32_t samples_size,
    const uint32_t clusters_size,
    const uint16_t features_size,
    const uint32_t yy_groups_size, 
    const float *samples,
    const uint32_t *mark_cpu, 
    const float *centroids,
    const uint32_t *groups, 
    const float *drifts,
    uint32_t *assignments, 
    float *bounds,
    float *second_min_dists,
    uint32_t *mark_cpu_number,
    uint32_t& changed_number_cpu,
    int32_t *mark_threads
    ) {

    double start_time = omp_get_wtime();  // Start timing

    omp_set_num_threads(omp_get_max_threads());
    changed_number_cpu = 0 ;
    
    #pragma omp parallel for schedule(static) reduction(+:changed_number_cpu)
    for (uint32_t i = 0; i < *mark_cpu_number; i++) {
        uint32_t sample = mark_cpu[i];
        float upper_bound = bounds[sample];
        uint32_t cluster = assignments[sample];
        uint32_t nearest = cluster;
        float min_dist = upper_bound;
        float second_min_dist = second_min_dists[sample];
        
        float min_dist_sq = min_dist * min_dist;
        float second_min_dist_sq = second_min_dist * second_min_dist;

        for (uint32_t i = 0; i < 32; i++) {
            int32_t mask = mark_threads[sample * 32 + i];

            if(mask == 0 && i < 31){
                uint32_t new_mask = 0;
                for(uint32_t j = 1 ; j < (32 - i ); j++){
                    new_mask = mark_threads[sample * 32 + i + j];
                    if(new_mask != 0){
                        i += j;
                        mask = new_mask;
                        break;
                    }
                }
            }

            while (mask != 0) {
                int first_bit_pos = __builtin_ffs(mask) - 1;  
                uint32_t c = i * 32 + first_bit_pos; 

                // compute distance without Neon
                // float dist = 0;
                // for (uint16_t f = 0; f < features_size; f++) {
                //     float d = samples[samples_size * f + sample] - centroids[c * features_size + f];
                //     dist += d * d;
                // }
                // dist = sqrt(dist);

                // neon or scalar
#if USE_NEON
                float dist_sq = distance_t_neon_squared<float>(samples, centroids + c * features_size, samples_size, features_size, sample);
#else
                float dist_sq = distance_t_squared<float>(samples, centroids + c * features_size, samples_size, features_size, sample);
#endif

                if (dist_sq < min_dist_sq) {
                    second_min_dist_sq = min_dist_sq;
                    second_min_dist = min_dist;
                    min_dist_sq = dist_sq;
                    min_dist = std::sqrt(dist_sq);
                    nearest = c;
                } else if (dist_sq < second_min_dist_sq) {
                    second_min_dist_sq = dist_sq;
                    second_min_dist = std::sqrt(dist_sq);
                }
                mask &= ~(1 << first_bit_pos);
            }
        }

        uint32_t nearest_group = groups[nearest];
        uint32_t previous_group = groups[cluster];
        bounds[samples_size * (1 + nearest_group) + sample] = second_min_dist;

        if (nearest_group != previous_group) {
            uint64_t gindex = static_cast<uint64_t>(samples_size) * (1 + previous_group) + sample;
            float pb = bounds[gindex];
            if (pb > upper_bound) {
                bounds[gindex] = upper_bound;
            }
        }
        bounds[sample] = min_dist;

        if (cluster != nearest) {
            assignments[sample] = nearest;
            changed_number_cpu += 1;
        }
    }
    double end_time = omp_get_wtime();  // End timing
    printf("(9-3-4) cpu execution time: %f ms\n", 1000*(end_time - start_time));

}

template <typename F>
void kubism_CPU_WB_HETD_batch4(
    const uint32_t samples_size,
    const uint16_t features_size,
    const float *samples,
    const uint32_t *mark_cpu, 
    const float *centroids,
    const uint32_t *groups, 
    uint32_t *assignments, 
    float *bounds,
    float *second_min_dists,
    uint32_t *mark_cpu_number,
    uint32_t& changed_number_cpu,
    int32_t *mark_threads
    ) {
 
    double start_time = omp_get_wtime(); 

    omp_set_num_threads(omp_get_max_threads());
    changed_number_cpu = 0 ;

    uint64_t total_batch4_calls = 0;
    uint64_t useful_distances = 0;
    
    #pragma omp parallel for schedule(static) reduction(+:changed_number_cpu, total_batch4_calls, useful_distances)
    for (uint32_t i = 0; i < *mark_cpu_number; i += 4) {
        uint32_t batch_size = std::min(4u, *mark_cpu_number - i);
        uint32_t s_indices[4];
        float u_bounds[4], m_dists[4], s_m_dists[4];
        uint32_t clusters[4], nearests[4];
        float m_dist_sqs[4], s_m_dist_sqs[4];

        for (uint32_t b = 0; b < 4; b++) {
            if (b < batch_size) {
                s_indices[b] = mark_cpu[i + b];
                u_bounds[b] = bounds[s_indices[b]];
                clusters[b] = assignments[s_indices[b]];
                nearests[b] = clusters[b];
                m_dists[b] = u_bounds[b];
                s_m_dists[b] = second_min_dists[s_indices[b]];
                m_dist_sqs[b] = m_dists[b] * m_dists[b];
                s_m_dist_sqs[b] = s_m_dists[b] * s_m_dists[b];
            } else {
                s_indices[b] = s_indices[0]; // Repeat s0
                m_dist_sqs[b] = FLT_MAX;
                s_m_dist_sqs[b] = FLT_MAX;
            }
        }

        for (uint32_t m = 0; m < 32; m++) {
            int32_t union_mask = 0;
            for (uint32_t b = 0; b < batch_size; b++) {
                union_mask |= mark_threads[s_indices[b] * 32 + m];
            }

            if (union_mask == 0) continue;

            while (union_mask != 0) {
                int first_bit_pos = __builtin_ffs(union_mask) - 1;  
                uint32_t c = m * 32 + first_bit_pos; 

                float b_dists_sq[4];
                // Pass pointer to current batch in the interleaved samples array
                distance_t_neon_batch4_squared<float>(samples + (i / 4) * features_size * 4, centroids + c * features_size, features_size, b_dists_sq);
                total_batch4_calls++;

                for (uint32_t b = 0; b < batch_size; b++) {
                    if (mark_threads[s_indices[b] * 32 + m] & (1 << first_bit_pos)) {
                        useful_distances++;
                        float dist_sq = b_dists_sq[b];
                        if (dist_sq < m_dist_sqs[b]) {
                            s_m_dist_sqs[b] = m_dist_sqs[b];
                            s_m_dists[b] = m_dists[b];
                            m_dist_sqs[b] = dist_sq;
                            m_dists[b] = std::sqrt(dist_sq);
                            nearests[b] = c;
                        } else if (dist_sq < s_m_dist_sqs[b]) {
                            s_m_dist_sqs[b] = dist_sq;
                            s_m_dists[b] = std::sqrt(dist_sq);
                        }
                    }
                }
                union_mask &= ~(1 << first_bit_pos);
            }
        }

        for (uint32_t b = 0; b < batch_size; b++) {
            uint32_t sample = s_indices[b];
            uint32_t nearest = nearests[b];
            uint32_t cluster = clusters[b];
            float min_dist = m_dists[b];
            float second_min_dist = s_m_dists[b];
            float upper_bound = u_bounds[b];

            uint32_t nearest_group = groups[nearest];
            uint32_t previous_group = groups[cluster];
            bounds[samples_size * (1 + nearest_group) + sample] = second_min_dist;

            if (nearest_group != previous_group) {
                uint64_t gindex = static_cast<uint64_t>(samples_size) * (1 + previous_group) + sample;
                float pb = bounds[gindex];
                if (pb > upper_bound) {
                    bounds[gindex] = upper_bound;
                }
            }
            bounds[sample] = min_dist;

            if (cluster != nearest) {
                assignments[sample] = nearest;
                changed_number_cpu += 1;
            }
        }
    }
    double end_time = omp_get_wtime(); 
    printf("(9-3-4) cpu execution time: %f ms\n", 1000*(end_time - start_time));

    uint64_t total_computed = total_batch4_calls * 4;
    uint64_t wasted = total_computed - useful_distances;
    double waste_ratio = total_computed > 0 ? (double)wasted / total_computed * 100.0 : 0.0;

    printf("[Batching Stats] Total Batch4 Calls: %lu\n", total_batch4_calls);
    printf("[Batching Stats] Total Individual Dists Computed: %lu\n", total_computed);
    printf("[Batching Stats] Useful Individual Dists (YY): %lu\n", useful_distances);
    printf("[Batching Stats] Wasted Dists (Batch Overhead): %lu (%.2f%%)\n", wasted, waste_ratio);
  }



template <typename F>
void kubism_CPU_HETD(
    const uint32_t samples_size,
    const uint32_t clusters_size,
    const uint16_t features_size,
    const uint32_t yy_groups_size,
    const float *samples,
    const uint32_t *mark_cpu, 
    const float *centroids,
    const uint32_t *groups, 
    const float *drifts,
    uint32_t *assignments, 
    float *bounds,
    float *second_min_dists,
    uint32_t *mark_cpu_number,
    uint32_t& changed_number_cpu
    ){

    double start_time = omp_get_wtime();  
    omp_set_num_threads(omp_get_max_threads());
    changed_number_cpu = 0 ;
    
    #pragma omp parallel for schedule(static) reduction(+:changed_number_cpu)
    for (uint32_t i = 0; i < *mark_cpu_number; i++) {
        uint32_t sample = mark_cpu[i];
        float upper_bound = bounds[sample];
        uint32_t cluster = assignments[sample];
        uint32_t nearest = cluster;
        float min_dist = upper_bound;
        float second_min_dist = second_min_dists[sample];
        
        float min_dist_sq = min_dist * min_dist;
        float second_min_dist_sq = second_min_dist * second_min_dist;
        uint32_t doffset = clusters_size * features_size;

        for (uint32_t c = 0; c < clusters_size; c++) {
            if (c == cluster) {
                continue;
            }
            uint32_t group = groups[c];
            if (group >= yy_groups_size){ 
                continue;
            }

            float lower_bound = bounds[static_cast<uint64_t>(samples_size) * (1 + group) + sample];
            // Local Filtering #1
            if (lower_bound >= upper_bound) {
                if (lower_bound < second_min_dist) {
                    second_min_dist = lower_bound;
                }
                continue;
            }
            lower_bound += drifts[group] - drifts[doffset + c];
            // Local Filtering #2
            if (lower_bound > second_min_dist) {
                continue;
            }
        
            // Calculate distance
#if USE_NEON
            float dist_sq = distance_t_neon_squared<float>(samples, centroids + c * features_size, samples_size, features_size, sample);
#else
            float dist_sq = distance_t_squared<float>(samples, centroids + c * features_size, samples_size, features_size, sample);
#endif
            // Update nearest cluster and distances
            if (dist_sq < min_dist_sq) {
                second_min_dist_sq = min_dist_sq;
                second_min_dist = min_dist;
                min_dist_sq = dist_sq;
                min_dist = std::sqrt(dist_sq);
                nearest = c;
            } else if (dist_sq < second_min_dist_sq) {
                second_min_dist_sq = dist_sq;
                second_min_dist = std::sqrt(dist_sq);
            }
            
        }

        uint32_t nearest_group = groups[nearest];
        uint32_t previous_group = groups[cluster];
        bounds[static_cast<uint64_t>(samples_size) * (1 + nearest_group) + sample] = second_min_dist;
        if (nearest_group != previous_group) {
            uint64_t gindex = static_cast<uint64_t>(samples_size) * (1 + previous_group) + sample;
            float pb = bounds[gindex];
            if (pb > upper_bound) {
                bounds[gindex] = upper_bound;
            }
        }
        bounds[sample] = min_dist;

        if (cluster != nearest) {
            assignments[sample] = nearest;
            
            changed_number_cpu += 1;
        }
    }
    #pragma omp barrier

    double end_time = omp_get_wtime();  // End timing
    printf("(9-4-4) cpu execution time: %f ms\n", 1000*(end_time - start_time));
}




template void kubism_CPU_WB_HETD<float>(
    const uint32_t samples_size,
    const uint32_t clusters_size,
    const uint16_t features_size,
    const uint32_t yy_groups_size,
    const float* samples,
    const uint32_t *mark_cpu, 
    const float* centroids,
    const uint32_t* groups, 
    const float* drifts,
    uint32_t* assignments, 
    float* bounds,
    float *second_min_dists,
    uint32_t *mark_cpu_number,
    uint32_t& changed_number_cpu,
    int32_t *mark_threads
);

template void distance_t_neon_batch4_squared<float>(
    const float *interleaved_samples,
    const float *centroids,
    uint16_t features_size,
    float *dists_sq
);

template void kubism_CPU_WB_HETD_batch4<float>(
    const uint32_t samples_size,
    const uint16_t features_size,
    const float *samples,
    const uint32_t *mark_cpu, 
    const float *centroids,
    const uint32_t *groups, 
    uint32_t *assignments, 
    float *bounds,
    float *second_min_dists,
    uint32_t *mark_cpu_number,
    uint32_t& changed_number_cpu,
    int32_t *mark_threads
);


template void kubism_CPU_HETD<float>(
    const uint32_t samples_size,
    const uint32_t clusters_size,
    const uint16_t features_size,
    const uint32_t yy_groups_size,
    const float *samples,
    const uint32_t *mark_cpu, 
    const float *centroids,
    const uint32_t *groups, 
    const float *drifts,
    uint32_t *assignments, 
    float *bounds,
    float *second_min_dists,
    uint32_t *mark_cpu_number,
    uint32_t& changed_number_cpu
);
