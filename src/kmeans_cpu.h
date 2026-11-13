/*
 * Modified by: Jaehyeok-Ryu (Futurism).
 * Based on comsys-lab/Kubism CPU declarations; modified in Futurism.
 * Futurism changes include distance, NEON and batch4 function interfaces.
 * The original implementation credit is retained below.
 * Pinned sources and file comparisons: docs/SOURCE_AUDIT.md.
 * Source attribution and license scope: docs/PROVENANCE.md.
 */

/********************************************
*
*Kubism - CPU Function
*Implemented by: Seondeok Kim
*
********************************************/

#ifndef KMEANS_CPU_H
#define KMEANS_CPU_H

#include <cstdint>
#include <atomic>



template <typename F>
float distance_t_squared (
    const float *v1,
    const float *v2,
    uint32_t v1_size,
    uint16_t features_size,
    uint32_t v1_index);

template <typename F>
float distance_t_neon_squared(
    const float *samples,   // v1
    const float *centroids, // v2
    uint32_t samples_size,  // v1_size
    uint16_t features_size, // features_size
    uint32_t sample         // v1_index
);


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
);

template <typename F>
void distance_t_neon_batch4_squared(
    const float *interleaved_samples,
    const float *centroids,
    uint16_t features_size,
    float *dists_sq
);

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
);

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
);



#endif
