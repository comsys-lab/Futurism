/*
 * Modified by: Jaehyeok-Ryu (Futurism).
 * Derived from KMCUDA through comsys-lab/Kubism; modified in Futurism.
 * Futurism changes include CUB compaction, CPU/GPU task partitioning,
 * raw CPU sample access, transposed GPU centroids and buffer/timing changes.
 * Pinned sources and file comparisons: docs/SOURCE_AUDIT.md.
 * Source attribution and license scope: docs/PROVENANCE.md.
 */

#define USE_CUB_PASSED_COMPACTION 1
#define USE_CUB_MARK_COMPACTION 1
#define USE_WB_HETD_BATCH4 0
#define USE_RAW_SAMPLES 1

#if USE_CUB_PASSED_COMPACTION || USE_CUB_MARK_COMPACTION
#include <cub/cub.cuh>
#endif

#include <cassert>
#include <cstdio>
#include <cfloat>
#include <cinttypes>
#include <algorithm>
#include <memory>
#include <time.h>
#include <curand_kernel.h>
#include <iostream>
#include <atomic>

#include <thread>
#include <chrono>
#include <iostream>

#include "private.h"
#include "metric_abstraction.h"
#include "tricks.cuh"

#include "kmeans_cpu.h"

#include "fp_abstraction.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


//#define SAMPLES_SIZE 1000000

#define BS_KMPP 1024
#define BS_AFKMC2_Q 512
#define BS_AFKMC2_R 512
#define BS_AFKMC2_MDT 512
#define BS_LL_ASS 128
#define BS_LL_CNT 256
#define BS_YY_INI 128
#define BS_YY_GFL 512
#define BS_YY_LFL 512
#define BS_HETD 512
#define BS_WB_HETD 512
// 1 thread block 이니까 128 -> 384, 256 -> 768 (1024 limit 있으므로 512는 안됨)

#define BLOCK_SIZE 1024  // for all the rest of the kernels
#define SHMEM_AFKMC2_RC 8191  // in float-s, the actual value is +1
#define SHMEM_AFKMC2_MT 8192

// #define USE_KUBISM_GPU_HETD_TILED 0

#define KUBISM_WB_HETD_USE_TRANSPOSED_CENTROIDS 1

#define HETD_DEBUG 0
#define HETD_CPU_GPU 1

#define YINYANG_GROUP_TOLERANCE 0.02
#define YINYANG_DRAFT_REASSIGNMENTS 0.11
#define YINYANG_REFRESH_EPSILON 1e-2

__device__ uint32_t d_changed_number;
__device__ uint32_t d_passed_number;

__constant__ uint32_t d_samples_size;
__constant__ uint32_t d_clusters_size;
__constant__ uint32_t d_yy_groups_size;
__constant__ int d_shmem_size;

// Experimental Baseline Execution Mode:
// Un-comment the next line to enable the Baseline Local Filter path.
// #define FUTURISM_EXECUTION_MODE_BASELINE


//////////////////////----------------------------------------------------------
// Device functions //----------------------------------------------------------
//////////////////////----------------------------------------------------------

template <typename F>
__device__ __forceinline__ F centroid_at(
    const F *__restrict__ centroids, uint32_t c, uint16_t f) {
#if KUBISM_WB_HETD_USE_TRANSPOSED_CENTROIDS
  return centroids[f * d_clusters_size + c];
#else
  return centroids[c * d_features_size + f];
#endif
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_plus_plus(
    const uint32_t offset, const uint32_t length, const uint32_t cc,
    const F *__restrict__ samples, const F *__restrict__ centroids,
    float *__restrict__ dists, atomic_float *__restrict__ dists_sum) {
  uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  float dist = 0;
  if (sample < length) {
    centroids += (cc - 1) * d_features_size;
    const uint32_t local_sample = sample + offset;
    if (_eq(samples[local_sample], samples[local_sample])) {
      dist = METRIC<M, F>::distance_t(
          samples, centroids, d_samples_size, local_sample);
    }
    float prev_dist;
    if (cc == 1 || dist < (prev_dist = dists[sample])) {
      dists[sample] = dist;
    } else {
      dist = prev_dist;
    }
  }
  dist = warpReduceSum(dist);
  if (threadIdx.x % 32 == 0) {
    atomicAdd(dists_sum, dist);
  }
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_afkmc2_calc_q_dists(
    const uint32_t offset, const uint32_t length, uint32_t c1_index,
    const F *__restrict__ samples, float *__restrict__ dists,
    atomic_float *__restrict__ dsum) {
  volatile uint64_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  float dist = 0;
  if (sample < length) {
    sample += offset;
    extern __shared__ float shmem_afkmc2[];
    auto c1 = reinterpret_cast<F*>(shmem_afkmc2);
    uint16_t size_each = dupper(d_features_size, static_cast<uint16_t>(blockDim.x));
    for (uint16_t i = size_each * threadIdx.x;
         i < min(size_each * (threadIdx.x + 1), d_features_size); i++) {
      c1[i] = samples[static_cast<uint64_t>(c1_index) * d_features_size + i];
    }
    __syncthreads();
    dist = METRIC<M, F>::distance_t(samples, c1, d_samples_size, sample);
    dist *= dist;
    dists[sample] = dist;
  }
  float sum = warpReduceSum(dist);
  if (threadIdx.x % 32 == 0) {
    atomicAdd(dsum, sum);
  }
}

__global__ void kmeans_afkmc2_calc_q(
    const uint32_t offset, const uint32_t length,
    float dsum, float *__restrict__ q) {
  volatile uint64_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= length) {
    return;
  }
  sample += offset;
  q[sample] = 1 / (2.f * d_samples_size) + q[sample] / (2 * dsum);
}

__global__ void kmeans_afkmc2_random_step(
    const uint32_t m, const uint64_t seed, const uint64_t seq,
    const float *__restrict__ q, uint32_t *__restrict__ choices,
    float *__restrict__ samples) {
  volatile uint32_t ti = blockIdx.x * blockDim.x + threadIdx.x;
  curandState_t state;
  curand_init(seed, ti, seq, &state);
  float part = curand_uniform(&state);
  if (ti < m) {
    samples[ti] = curand_uniform(&state);
  }
  float accum = 0, corr = 0;
  bool found = false;
  __shared__ float shared_q[SHMEM_AFKMC2_RC + 1];
  int32_t *all_found = reinterpret_cast<int32_t*>(shared_q + SHMEM_AFKMC2_RC);
  *all_found = blockDim.x;
  const uint32_t size_each = dupper(
      static_cast<unsigned>(SHMEM_AFKMC2_RC), blockDim.x);
  for (uint32_t sample = 0; sample < d_samples_size; sample += SHMEM_AFKMC2_RC) {
    __syncthreads();
    if (*all_found == 0) {
      return;
    }
    for (uint32_t i = 0, si = threadIdx.x * size_each;
         i < size_each && (si = threadIdx.x * size_each + i) < SHMEM_AFKMC2_RC
         && (sample + si) < d_samples_size;
         i++) {
      shared_q[si] = q[sample + si];
    }
    __syncthreads();
    if (!found) {
      int i = 0;
      for (; i < SHMEM_AFKMC2_RC && accum < part && sample + i < d_samples_size;
           i++) {
        // Kahan summation with inverted c
        float y = _add(corr, shared_q[i]);
        float t = accum + y;
        corr = y - (t - accum);
        accum = t;
      }
      if (accum >= part) {
        if (ti < m) {
          choices[ti] = sample + i - 1;
        }
        found = true;
        atomicSub(all_found, 1);
      }
    }
  }
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_afkmc2_min_dist(
    const uint32_t m, const uint32_t k, const F *__restrict__ samples,
    const uint32_t *__restrict__ choices, const F *__restrict__ centroids,
    float *__restrict__ min_dists) {
  uint32_t chi = blockIdx.x * blockDim.x + threadIdx.x;
  if (chi >= m) {
    return;
  }
  float min_dist = FLT_MAX;
  for (uint32_t c = 0; c < k; c++) {
    float dist = METRIC<M, F>::distance_t(
        samples, centroids + c * d_features_size, d_samples_size, choices[chi]);
    if (dist < min_dist) {
      min_dist = dist;
    }
  }
  min_dists[chi] = min_dist * min_dist;
}

// min_dists must be set to FLT_MAX or +inf or NAN!
template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_afkmc2_min_dist_transposed(
    const uint32_t m, const uint32_t k, const F *__restrict__ samples,
    const uint32_t *__restrict__ choices, const F *__restrict__ centroids,
    float *__restrict__ min_dists) {
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ float shared_min_dists[];
  uint32_t size_each = dupper(m, blockDim.x);
  for (uint32_t i = size_each * threadIdx.x;
       i < min(size_each * (threadIdx.x + 1), m);
       i++) {
    shared_min_dists[i] = FLT_MAX;
  }
  __syncthreads();
  for (uint32_t chi = 0; chi < m; chi++) {
    float dist = FLT_MAX;
    if (c < k) {
      dist = METRIC<M, F>::distance_t(
          samples, centroids + c * d_features_size, d_samples_size, choices[chi]);
    }
    float warp_min = warpReduceMin(dist);
    warp_min *= warp_min;
    if (threadIdx.x % 32 == 0 && c < k) {
      atomicMin(shared_min_dists + chi, warp_min);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (uint32_t chi = 0; chi < m; chi++) {
      atomicMin(min_dists + chi, shared_min_dists[chi]);
    }
  }
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_assign_lloyd_smallc(
    const uint32_t offset, const uint32_t length, const F *__restrict__ samples,
    const F *__restrict__ centroids, uint32_t *__restrict__ assignments_prev,
    uint32_t * __restrict__ assignments) {
  using HF = typename HALF<F>::type;
  uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= length) {
    return;
  }
  HF min_dist = _fmax<HF>();
  uint32_t nearest = UINT32_MAX;
  extern __shared__ float _shared_samples[];
  F *shared_samples = reinterpret_cast<F *>(_shared_samples);
  F *shared_centroids = shared_samples + blockDim.x * d_features_size;
  const uint32_t cstep = (d_shmem_size - blockDim.x * d_features_size) /
      (d_features_size + 1);
  F *csqrs = shared_centroids + cstep * d_features_size;
  const uint32_t size_each = cstep /
      min(blockDim.x, length - blockIdx.x * blockDim.x) + 1;
  const uint32_t local_sample = sample + offset;
  bool insane = _neq(samples[local_sample], samples[local_sample]);
  const uint32_t soffset = threadIdx.x * d_features_size;
  if (!insane) {
    for (uint64_t f = 0; f < d_features_size; f++) {
      shared_samples[soffset + f] = samples[f * d_samples_size + local_sample];
    }
  }

  for (uint32_t gc = 0; gc < d_clusters_size; gc += cstep) {
    uint32_t coffset = gc * d_features_size;
    __syncthreads();
    for (uint32_t i = 0; i < size_each; i++) {
      uint32_t ci = threadIdx.x * size_each + i;
      uint32_t local_offset = ci * d_features_size;
      uint32_t global_offset = coffset + local_offset;
      if (global_offset < d_clusters_size * d_features_size && ci < cstep) {
        csqrs[ci] = METRIC<M, F>::sum_squares(
            centroids + global_offset, shared_centroids + local_offset);
      }
    }
    __syncthreads();
    if (insane) {
      continue;
    }
    for (uint32_t c = gc; c < gc + cstep && c < d_clusters_size; c++) {
      F product = _const<F>(0), corr = _const<F>(0);
      coffset = (c - gc) * d_features_size;
      #pragma unroll 4
      for (int f = 0; f < d_features_size; f++) {
        F y = _fma(corr, shared_samples[soffset + f], shared_centroids[coffset + f]);
        F t = _add(product, y);
        corr = _sub(y, _sub(t, product));
        product = t;
      }
      HF dist = METRIC<M, F>::distance(_const<F>(0), csqrs[c - gc], product);
      if (_lt(dist, min_dist)) {
        min_dist = dist;
        nearest = c;
      }
    }
  }
  if (nearest == UINT32_MAX) {
    if (!insane) {
      printf("CUDA kernel kmeans_assign: nearest neighbor search failed for " "sample %" PRIu32 "\n", sample);
      return;
    } else {
      nearest = d_clusters_size;
    }
  }
  uint32_t ass = assignments[sample];
  assignments_prev[sample] = ass;
  if (ass != nearest) {
    assignments[sample] = nearest;
    atomicAggInc(&d_changed_number);
  }
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_assign_lloyd(
    const uint32_t offset, const uint32_t length, const F *__restrict__ samples,
    const F *__restrict__ centroids, uint32_t *__restrict__ assignments_prev,
    uint32_t * __restrict__ assignments) {
  using HF = typename HALF<F>::type;
  uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= length) {
    return;
  }
  HF min_dist = _fmax<HF>();
  uint32_t nearest = UINT32_MAX;
  extern __shared__ float _shared_centroids[];
  F *shared_centroids = reinterpret_cast<F *>(_shared_centroids);
  const uint32_t cstep = d_shmem_size / (d_features_size + 1);
  F *csqrs = shared_centroids + cstep * d_features_size;
  const uint32_t size_each = cstep /
      min(blockDim.x, length - blockIdx.x * blockDim.x) + 1;
  const uint32_t local_sample = sample + offset;
  bool insane = _neq(samples[local_sample], samples[local_sample]);

  for (uint32_t gc = 0; gc < d_clusters_size; gc += cstep) {
    uint32_t coffset = gc * d_features_size;
    __syncthreads();
    for (uint32_t i = 0; i < size_each; i++) {
      uint32_t ci = threadIdx.x * size_each + i;
      uint32_t local_offset = ci * d_features_size;
      uint32_t global_offset = coffset + local_offset;
      if (global_offset < d_clusters_size * d_features_size && ci < cstep) {
        csqrs[ci] = METRIC<M, F>::sum_squares(
            centroids + global_offset, shared_centroids + local_offset);
      }
    }
    __syncthreads();
    if (insane) {
      continue;
    }
    for (uint32_t c = gc; c < gc + cstep && c < d_clusters_size; c++) {
      F product = _const<F>(0), corr = _const<F>(0);
      coffset = (c - gc) * d_features_size;
      #pragma unroll 4
      for (uint64_t f = 0; f < d_features_size; f++) {
        F y = _fma(corr,
                   samples[static_cast<uint64_t>(d_samples_size) * f + local_sample],
                   shared_centroids[coffset + f]);
        F t = _add(product, y);
        corr = _sub(y, _sub(t, product));
        product = t;
      }
      HF dist = METRIC<M, F>::distance(_const<F>(0), csqrs[c - gc], product);
      if (_lt(dist, min_dist)) {
        min_dist = dist;
        nearest = c;
      }
    }
  }
  if (nearest == UINT32_MAX) {
    if (!insane) {
      printf("CUDA kernel kmeans_assign: nearest neighbor search failed for " "sample %" PRIu32 "\n", sample);
      return;
    } else {
      nearest = d_clusters_size;
    }
  }
  uint32_t ass = assignments[sample];
  assignments_prev[sample] = ass;
  if (ass != nearest) {
    assignments[sample] = nearest;
    atomicAggInc(&d_changed_number);
  }
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_adjust(
    const uint32_t coffset, const uint32_t length,
    const F *__restrict__ samples,
    const uint32_t *__restrict__ assignments_prev,
    const uint32_t *__restrict__ assignments,
    F *__restrict__ centroids, uint32_t *__restrict__ ccounts) {
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= length) {
    return;
  }
  c += coffset;
  uint32_t my_count = ccounts[c];
  {
    F fmy_count = _const<F>(my_count);
    centroids += c * d_features_size;
    for (int f = 0; f < d_features_size; f++) {
      centroids[f] = _mul(centroids[f], fmy_count);
    }
  }
  extern __shared__ uint32_t ass[];
  int step = d_shmem_size / 2;
  F corr = _const<F>(0);
  for (uint32_t sbase = 0; sbase < d_samples_size; sbase += step) {
    __syncthreads();
    if (threadIdx.x == 0) {
      int pos = sbase;
      for (int i = 0; i < step && sbase + i < d_samples_size; i++) {
        ass[2 * i] = assignments[pos + i];
        ass[2 * i + 1] = assignments_prev[pos + i];
      }
    }
    __syncthreads();
    for (int i = 0; i < step && sbase + i < d_samples_size; i++) {
      uint32_t this_ass = ass[2 * i];
      uint32_t  prev_ass = ass[2 * i + 1];
      int sign = 0;
      if (prev_ass == c && this_ass != c) {
        sign = -1;
        my_count--;
      } else if (prev_ass != c && this_ass == c) {
        sign = 1;
        my_count++;
      }
      if (sign != 0) {
        F fsign = _const<F>(sign);
        #pragma unroll 4
        for (uint64_t f = 0; f < d_features_size; f++) {
          F centroid = centroids[f];
          F y = _fma(corr,
                     samples[static_cast<uint64_t>(d_samples_size) * f + sbase + i],
                     fsign);
          F t = _add(centroid, y);
          corr = _sub(y, _sub(t, centroid));
          centroids[f] = t;
        }
      }
    }
  }
  // my_count can be 0 => we get NaN with L2 and never use this cluster again
  // this is a feature, not a bug
  METRIC<M, F>::normalize(my_count, centroids);
  ccounts[c] = my_count;
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_init(
    const uint32_t offset, const uint32_t length, const F *__restrict__ samples,
    const F *__restrict__ centroids, const uint32_t *__restrict__ assignments,
    const uint32_t *__restrict__ groups, float *__restrict__ volatile bounds) {
  volatile uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= length) {
    return;
  }
  for (uint32_t i = 0; i < d_yy_groups_size + 1; i++) {
    bounds[static_cast<uint64_t>(length) * i + sample] = FLT_MAX;
  }

  uint32_t nearest = assignments[sample];
  extern __shared__ float shared_memory[];
  F *volatile shared_centroids = reinterpret_cast<F*>(shared_memory);
  const uint32_t cstep = d_shmem_size / d_features_size;  //  96
  const uint32_t size_each = cstep / min(blockDim.x, length - blockIdx.x * blockDim.x) + 1;


  for (uint32_t gc = 0; gc < d_clusters_size; gc += cstep) {
    uint32_t coffset = gc * d_features_size;
    __syncthreads();
    for (uint32_t i = 0; i < size_each; i++) {
      uint32_t ci = threadIdx.x * size_each + i;
      uint32_t local_offset = ci * d_features_size;
      uint32_t global_offset = coffset + local_offset;
      if (global_offset < d_clusters_size * d_features_size && ci < cstep) {
        #pragma unroll 4
        for (int f = 0; f < d_features_size; f++) {
          shared_centroids[local_offset + f] = centroids[global_offset + f];
        }
      }
    }
    __syncthreads();

    for (uint32_t c = gc; c < gc + cstep && c < d_clusters_size; c++) {
      uint32_t group = groups[c];
      if (group >= d_yy_groups_size) {
        // this may happen if the centroid is insane (NaN)
        continue;
      }
      float dist = METRIC<M, F>::distance_t(
          samples, shared_centroids + (c - gc) * d_features_size,
          d_samples_size, sample + offset);
      if (c != nearest) {
        uint64_t gindex = static_cast<uint64_t>(length) * (1 + group) + sample;
        if (dist < bounds[gindex]) {
          bounds[gindex] = dist;
        }
      } else {
        bounds[sample] = dist;
      }
    }
  }
  
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_calc_drifts(
    const uint32_t offset, const uint32_t length,
    const F *__restrict__ centroids, F *__restrict__ drifts) {
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= length) {
    return;
  }
  c += offset;
  uint32_t coffset = c * d_features_size;
  (reinterpret_cast<float *>(drifts))[d_clusters_size * d_features_size + c] =
      METRIC<M, F>::distance(centroids + coffset, drifts + coffset);
}

__global__ void kmeans_yy_find_group_max_drifts(
    const uint32_t offset, const uint32_t length,
    const uint32_t *__restrict__ groups, float *__restrict__ drifts) {
  uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= length) {
    return;
  }
  group += offset;
  const uint32_t doffset = d_clusters_size * d_features_size;
  const uint32_t step = d_shmem_size / 2;
  const uint32_t size_each = d_shmem_size /
      (2 * min(blockDim.x, length - blockIdx.x * blockDim.x));
  extern __shared__ uint32_t shmem[];
  float *cd = (float *)shmem;
  uint32_t *cg = shmem + step;
  float my_max = -FLT_MAX;
  for (uint32_t offset = 0; offset < d_clusters_size; offset += step) {
    __syncthreads();
    for (uint32_t i = 0; i < size_each; i++) {
      uint32_t local_offset = threadIdx.x * size_each + i;
      uint32_t global_offset = offset + local_offset;
      if (global_offset < d_clusters_size && local_offset < step) {
        cd[local_offset] = drifts[doffset + global_offset];
        cg[local_offset] = groups[global_offset];
      }
    }
    __syncthreads();
    for (uint32_t i = 0; i < step; i++) {
      if (cg[i] == group) {
        float d = cd[i];
        if (my_max < d) {
          my_max = d;
        }
      }
    }
  }
  drifts[group] = my_max;
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_global_filter( // group filter
    const uint32_t offset, const uint32_t length, const F *__restrict__ samples,
    const F *__restrict__ centroids, const uint32_t *__restrict__ groups,
    const float *__restrict__ drifts, const uint32_t *__restrict__ assignments,
    uint32_t *__restrict__ assignments_prev, float *__restrict__ bounds,
  uint32_t *__restrict__ passed
#if USE_CUB_PASSED_COMPACTION
  , uint8_t *__restrict__ passed_flags
#endif
  ) {
  volatile uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= length) {
    return;
  }
  uint32_t cluster = assignments[sample];
  assignments_prev[sample] = cluster;

  float upper_bound = bounds[sample];
  uint32_t doffset = d_clusters_size * d_features_size;
  float cluster_drift = drifts[doffset + cluster];

  upper_bound += cluster_drift;
  float min_lower_bound = FLT_MAX;

  /* Find Global lower bound */
  for (uint32_t g = 0; g < d_yy_groups_size; g++) { // filtering at centroid group level
    uint64_t gindex = static_cast<uint64_t>(length) * (1 + g) + sample;
    float lower_bound = bounds[gindex] - drifts[g];
    bounds[gindex] = lower_bound;
    if (lower_bound < min_lower_bound) {
      min_lower_bound = lower_bound;
    }
  }

  // Group filter #1 // 
  if (min_lower_bound >= upper_bound) {
    bounds[sample] = upper_bound;

    return;
  }

  /* Upper bound adjust */
  upper_bound = 0;
  upper_bound = METRIC<M, F>::distance_t( samples, centroids + cluster * d_features_size, d_samples_size, sample + offset);
  bounds[sample] = upper_bound;
  
  // Group filter #2 // 
  if (min_lower_bound >= upper_bound) {
   return;  
  }

  // d'oh!
#if USE_CUB_PASSED_COMPACTION
  passed_flags[sample] = 1;
#else
  passed[atomicAggInc(&d_passed_number)] = sample;
#endif

  
}

//------------------------------------------------------------------------------------------------------

template <KMCUDADistanceMetric M, typename F>
__global__ void kubism_warp_balancing(
    const uint32_t length,
    const uint32_t *__restrict__ passed,
    const uint32_t *__restrict__ groups, const float *__restrict__ drifts,
    uint32_t *__restrict__ assignments, float *__restrict__ bounds, float *__restrict__ second_min_dists,
    int32_t *__restrict__ mark_threads){
  
  volatile uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;

  if(sample >= d_passed_number){ 
    return;
  }
 
  sample = passed[sample];

  float upper_bound = bounds[sample];
  uint32_t cluster = assignments[sample];
  float second_min_dist = FLT_MAX;
  uint32_t doffset = d_clusters_size * d_features_size; 

  uint32_t bitmask_register = 0 ;  // form 32 bit metadata

  for(uint32_t c = 0 ; c < d_clusters_size ; c++){  
    int mask_index = (c) / 32;     // metadata offset
    int bit_position = ( c) % 32;  // bit inside metadata

    /* Flush the 32-bit metadata once it is full or at the last centroid */
    bool should_store = ((c+1) % 32 == 0 || c == d_clusters_size - 1);  

    if(c == cluster) {
        if(should_store){
            mark_threads[sample * 32 + (mask_index)] = bitmask_register;
            bitmask_register = 0;
        }     
        continue;
    }
    uint32_t group = groups[c];
    if (group >= d_yy_groups_size) {
        if(should_store){
            mark_threads[sample * 32 + (mask_index)] = bitmask_register;
            bitmask_register = 0;
        }     
    continue;

    }
    float lower_bound = bounds[static_cast<uint64_t>(length) * (1 + group) + sample];

    /* Local Filtering #1 */
    if (lower_bound >= upper_bound) {
        if (lower_bound < second_min_dist) {
            second_min_dist = lower_bound;
        }
        if(should_store){
            mark_threads[sample * 32 + (mask_index)] = bitmask_register;
            bitmask_register = 0;
        }     
        continue;
    }

    lower_bound += drifts[group] - drifts[doffset + c];
    /* Local Filtering #2 */
    if (second_min_dist < lower_bound) {
        if(should_store){
            mark_threads[sample * 32 + (mask_index)] = bitmask_register;
            bitmask_register = 0;
        }     
        continue;
    }
    // Set bit to 1 if current centroid needs distance computation
    bitmask_register |= (1 << bit_position);

    if(should_store){
        mark_threads[sample * 32 + (mask_index)] = bitmask_register;
        bitmask_register = 0;
    }     
    }
  second_min_dists[sample] = second_min_dist; 
}
 //----------------------------------------------------------------------------------------------

template <KMCUDADistanceMetric M, typename F>
__global__ void kubism_clustering_reordering(
    const uint32_t length, const uint32_t *__restrict__ passed,
    const uint32_t *__restrict__ groups, const float *__restrict__ drifts,
    uint32_t *__restrict__ assignments, float *__restrict__ bounds, float *__restrict__ second_min_dists,
    uint32_t *calculate_sum){
  
  volatile uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;

  if(sample >= d_passed_number){ 
    return;
  }
 
  sample = passed[sample];

  float upper_bound = bounds[sample];
  uint32_t cluster = assignments[sample];
  float second_min_dist = FLT_MAX;
  uint32_t doffset = d_clusters_size * d_features_size;
  int16_t index = 0; 

  for(uint32_t c = 0 ; c < d_clusters_size ; c++){   
    
    if(c == cluster) {
      continue;
    }
    uint32_t group = groups[c];
    if (group >= d_yy_groups_size) {
      continue;
    }
    float lower_bound = bounds[static_cast<uint64_t>(length) * (1 + group) + sample];
    /* Local Filtering #1 */
    if (lower_bound >= upper_bound) {
      if (lower_bound < second_min_dist) {
        second_min_dist = lower_bound;
      }
      continue;
    }
    lower_bound += drifts[group] - drifts[doffset + c];
    /* Local Filtering #2 */
    if (second_min_dist < lower_bound) {
      continue;
    }
    index++; // Count the number of per-sample distance calculations
  }
    atomicAdd(calculate_sum, index); // Accumulate per-sample count into total sum

  second_min_dists[sample] = second_min_dist; 

}

//-----------------------------------------------------------------------------------------------

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yinyang_local_filter(
    const uint32_t length, const float *__restrict__ samples,
    const uint32_t *__restrict__ passed, const float *__restrict__ centroids,
    const uint32_t *__restrict__ groups, const float *__restrict__ drifts,
    uint32_t *__restrict__ assignments, float *__restrict__ bounds,
    float *__restrict__ second_min_dists) {

  volatile uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;

  if (sample >= d_passed_number){
    return;
  }

  sample = passed[sample];

  float upper_bound = bounds[sample];
  uint32_t cluster = assignments[sample];
  uint32_t nearest = cluster;
  float min_dist = upper_bound;
  float second_min_dist = FLT_MAX;
  uint32_t doffset = d_clusters_size * d_features_size;


  for (uint32_t c = 0; c < d_clusters_size; c++) {
    if (c == cluster) {
      continue;
    }

    uint32_t group = groups[c];
    if (group >= d_yy_groups_size) {
      // this may happen if the centroid is insane (NaN)
      continue;
    }

    float lower_bound = bounds[static_cast<uint64_t>(length) * (1 + group) + sample];
    /* Local Filter Filtering 1 */
    if (lower_bound >= upper_bound) {
      if (lower_bound < second_min_dist) {
        second_min_dist = lower_bound;
      }
      continue;
    }
    lower_bound += drifts[group] - drifts[doffset + c];
    /* Local Filter Filtering 2 */
    if (second_min_dist < lower_bound) {
      continue;
    }

    // Use shared memory for distance calculation
    float dist = 0;
    #pragma unroll 4
    for (uint16_t f = 0; f < d_features_size; ++f) {
      float d = samples[d_samples_size * f + sample] - centroids[(c) * d_features_size + f];
      dist += d * d;
    }
    float dist1 = sqrt(dist);

    if (dist1 < min_dist) {
      second_min_dist = min_dist;
      min_dist = dist1;
      nearest = c;
    } else if (dist1 < second_min_dist) {
      second_min_dist = dist1;
    }
  }

  uint32_t nearest_group = groups[nearest];
  uint32_t previous_group = groups[cluster];
  bounds[static_cast<uint64_t>(length) * (1 + nearest_group) + sample] = second_min_dist;

  if (nearest_group != previous_group) {
    uint64_t gindex = static_cast<uint64_t>(length) * (1 + previous_group) + sample;
    float pb = bounds[gindex];
    if (pb > upper_bound) {
      bounds[gindex] = upper_bound;
    }
  }
  
  bounds[sample] = min_dist;

  if (cluster != nearest) {
    assignments[sample] = nearest;
    atomicAggInc(&d_changed_number);
  }
}


//--------------------------------------------------------------------------------------------------------------------------
template <KMCUDADistanceMetric M, typename F>
__global__ void kubism_CPU_GPU_task_partition(
    const uint32_t *__restrict__ passed,
    uint32_t *__restrict__ mark_cpu, uint32_t *__restrict__ mark_gpu, float current_skip_ratio,
    uint32_t *d_mark_cpu_number, uint32_t *d_mark_gpu_number
#if USE_CUB_MARK_COMPACTION
    , uint8_t *__restrict__ mark_cpu_flags, uint8_t *__restrict__ mark_gpu_flags
#endif
    ){

  int32_t pass_idx = blockIdx.x * blockDim.x + threadIdx.x; 
  if (pass_idx >= d_passed_number) {
      return;  
  }

#if USE_CUB_MARK_COMPACTION
  bool selected = false;
  bool use_cpu = false;
#define MARK_CPU() do { selected = true; use_cpu = true; } while (0)
#define MARK_GPU() do { selected = true; use_cpu = false; } while (0)

#else

#define MARK_CPU() do { mark_cpu[atomicAggInc(d_mark_cpu_number)] = sample; } while (0)
#define MARK_GPU() do { mark_gpu[atomicAggInc(d_mark_gpu_number)] = sample; } while (0)
#endif
uint32_t sample = passed[pass_idx]; 

  /* Data point distribution between CPU and GPU varies by dataset. */
    // clustering reordering + warp balancing + HETD

    #if HETD_DEBUG
    if (current_skip_ratio < 1){
      if(HETD_CPU_GPU){
          MARK_CPU();
      }else{
          MARK_GPU();
      }
    } 

    #else

    float sr = current_skip_ratio;
    uint32_t threshold;

    // Futurism thresholds for 1,000,000 samples and 128 features.
    // Values from the original run_optimized_measurements.py configuration.
    if      (sr >= 0.95f) threshold = 679000;
    else if (sr >= 0.90f) threshold = 591000;
    else if (sr >= 0.80f) threshold = 519000;
    else if (sr >= 0.75f) threshold = 502000;
    else if (sr >= 0.65f) threshold = 497000;
    else if (sr >= 0.60f) threshold = 510000;
    else if (sr >= 0.50f) threshold = 520000;
    else                  threshold = 518000;

    if (sample < threshold) {
      MARK_CPU();
    } else {
      MARK_GPU();
    }

    #endif
    
#if USE_CUB_MARK_COMPACTION
  if (selected) {
    mark_cpu_flags[pass_idx] = use_cpu ? 1 : 0;
    mark_gpu_flags[pass_idx] = use_cpu ? 0 : 1;
  } else {
    mark_cpu_flags[pass_idx] = 0;
    mark_gpu_flags[pass_idx] = 0;
  }
#endif

#undef MARK_CPU
#undef MARK_GPU
}

//--------------------------------------------------------------------------------------------------------------------------
template <KMCUDADistanceMetric M, typename F>
__global__ void kubism_GPU_WB_HETD(
    const uint32_t length, const float *__restrict__ samples,
    const uint32_t *__restrict__ mark_gpu, 
    const float *__restrict__ centroids,
    const uint32_t *__restrict__ groups,
    uint32_t *__restrict__ assignments, float *__restrict__ bounds, float *__restrict__ second_min_dists, uint32_t *d_mark_gpu_number, int32_t *__restrict__ mark_threads) {

  volatile uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= *d_mark_gpu_number) {
    return;
  }

  sample = mark_gpu[sample];

  float upper_bound = bounds[sample];
  uint32_t cluster = assignments[sample];
  uint32_t nearest = cluster;
  float min_dist = upper_bound;
  float second_min_dist = second_min_dists[sample];

  // Iterate over 32-bit blocks of centroid metadata
  for(uint32_t i = 0 ; i < 32 ; i++){
    int32_t mask = mark_threads[sample * 32 + i]; 
    // Skip over zero metadata quickly
    if(mask == 0 && i < 31){
      uint32_t new_mask = 0;
      for(uint32_t j = 1 ; j < (32-i); j++){ 
        new_mask = mark_threads[sample * 32 + i + j];
        if(new_mask != 0){
            i += j;
            mask = new_mask;
            break;
        }
      }
    }
    
    // Process all active centroids in current bitmask
    while(mask!= 0){
      int first_bit_pos = __ffs(mask) - 1; 
      uint32_t c = i * 32 + (first_bit_pos); 

      float dist = 0;
      #pragma unroll 4
      for (uint16_t f = 0; f < d_features_size; ++f) {
          float d = samples[d_samples_size * f + sample] - centroid_at(centroids, c, f);
          dist += d * d;
      }
      float dist_1 = sqrt(dist); // Compute Euclidean distance

      // Update nearest and second nearest distances
      if (dist_1 < min_dist) {
          second_min_dist = min_dist;
          min_dist = dist_1;
          nearest = c;
      } else if (dist_1 < second_min_dist) {
          second_min_dist = dist_1;
      }
      mask &= ~(1 << first_bit_pos);  // Clear the processed bit
    }
  }
  // Update second min distance bound for the nearest group
  uint32_t nearest_group = groups[nearest];
  uint32_t previous_group = groups[cluster];
  bounds[static_cast<uint64_t>(length) * (1 + nearest_group) + sample] = second_min_dist;
  // Update previous group bound if necessary
  if (nearest_group != previous_group) {
    uint64_t gindex = static_cast<uint64_t>(length) * (1 + previous_group) + sample;
    float pb = bounds[gindex];
    if (pb > upper_bound) {
      bounds[gindex] = upper_bound;
    }
  }
  bounds[sample] = min_dist;
  // Reassign if cluster changed
  if (cluster != nearest) {
    assignments[sample] = nearest;
    atomicAggInc(&d_changed_number);
  }
}

//--------------------------------------------------------------------------------------------------------------------------
template <KMCUDADistanceMetric M, typename F>
__global__ void kubism_GPU_HETD(
    const uint32_t offset, const uint32_t length, const float *__restrict__ samples,
    const uint32_t *__restrict__ mark_gpu, 
    const uint32_t *__restrict__ passed, 
    const float *__restrict__ centroids,
    const uint32_t *__restrict__ groups, const float *__restrict__ drifts,
    uint32_t *__restrict__ assignments, float *__restrict__ bounds, float *__restrict__ second_min_dists, uint32_t *d_mark_gpu_number,
    uint32_t *partition_threshold
   ) {

   volatile uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;

  if (sample >= *d_mark_gpu_number) {
    return;
  }

  sample = mark_gpu[sample];

  float upper_bound = bounds[sample];
  uint32_t cluster = assignments[sample];
  uint32_t nearest = cluster;
  float min_dist = upper_bound;
  float second_min_dist = second_min_dists[sample];
  uint32_t doffset = d_clusters_size * d_features_size;

  for (uint32_t c = 0; c < d_clusters_size ; c++) {
    if (c == cluster) {
        continue;
    }
    uint32_t group = groups[c];
    if (group >= d_yy_groups_size) {
      // this may happen if the centroid is insane (NaN)
      continue;
    }
    float lower_bound = bounds[static_cast<uint64_t>(length) * (1 + group) + sample];
    if (lower_bound >= upper_bound) {
      if (lower_bound < second_min_dist) {
        second_min_dist = lower_bound;
      }
      continue;
    }
    lower_bound += drifts[group] - drifts[doffset + c];
    if (second_min_dist < lower_bound) {
      continue;
    }

    /* Distance Calculation */
    float dist = 0;
    #pragma unroll 4
    for (uint16_t f = 0; f < d_features_size; ++f) {
      float d = samples[d_samples_size * f + sample] - centroids[c * d_features_size + f];
        dist += d * d;
    }
    float dist_1 = sqrt(dist);

    if (dist_1 < min_dist) {
      second_min_dist = min_dist;
      min_dist = dist_1;
      nearest = c;
    } else if (dist_1 < second_min_dist) {
      second_min_dist = dist_1;
    }
  }

  uint32_t nearest_group = groups[nearest];
  uint32_t previous_group = groups[cluster];
  bounds[static_cast<uint64_t>(length) * (1 + nearest_group) + sample] = second_min_dist;
  
  if (nearest_group != previous_group) {
    uint64_t gindex = static_cast<uint64_t>(length) * (1 + previous_group) + sample;
    float pb = bounds[gindex];
    if (pb > upper_bound) {
      bounds[gindex] = upper_bound;
    }
  }
  bounds[sample] = min_dist;
 
  if (cluster != nearest) {
    assignments[sample] = nearest;
    atomicAggInc(&d_changed_number);
  }
}

//--------------------------------------------------------------------------------------------------------------------------
template <KMCUDADistanceMetric M, typename F>
__global__ void kubism_GPU_HETD_tiled(
    const uint32_t offset, const uint32_t length, const float *__restrict__ samples,
    const uint32_t *__restrict__ mark_gpu,
    const uint32_t *__restrict__ passed,
    const float *__restrict__ centroids,
    const uint32_t *__restrict__ groups, const float *__restrict__ drifts,
    uint32_t *__restrict__ assignments, float *__restrict__ bounds, float *__restrict__ second_min_dists, uint32_t *d_mark_gpu_number,
    uint32_t *partition_threshold
   ) {

  volatile uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = (sample < *d_mark_gpu_number);
  sample = active ? mark_gpu[sample] : 0;

  float upper_bound = 0;
  uint32_t cluster = 0;
  uint32_t nearest = 0;
  float min_dist = 0;
  float second_min_dist = 0;
  if (active) {
    upper_bound = bounds[sample];
    cluster = assignments[sample];
    nearest = cluster;
    min_dist = upper_bound;
    second_min_dist = second_min_dists[sample];
  }
  uint32_t doffset = d_clusters_size * d_features_size;

  const uint32_t shmem_reserve = 0;
  const uint32_t available = (d_shmem_size > shmem_reserve) ? (d_shmem_size - shmem_reserve) : 0;
  const uint32_t tileC = (d_features_size == 0) ? 0 : (available / d_features_size);

  if (tileC == 0) {
    if (active) {
      for (uint32_t c = 0; c < d_clusters_size; c++) {
      if (c == cluster) {
        continue;
      }
      uint32_t group = groups[c];
      if (group >= d_yy_groups_size) {
        continue;
      }
      float lower_bound = bounds[static_cast<uint64_t>(length) * (1 + group) + sample];
      if (lower_bound >= upper_bound) {
        if (lower_bound < second_min_dist) {
          second_min_dist = lower_bound;
        }
        continue;
      }
      lower_bound += drifts[group] - drifts[doffset + c];
      if (second_min_dist < lower_bound) {
        continue;
      }

      float dist = 0;
      #pragma unroll 4
      for (uint16_t f = 0; f < d_features_size; ++f) {
        float d = samples[d_samples_size * f + sample] - centroids[c * d_features_size + f];
        dist += d * d;
      }
      float dist_1 = sqrt(dist);

        if (dist_1 < min_dist) {
          second_min_dist = min_dist;
          min_dist = dist_1;
          nearest = c;
        } else if (dist_1 < second_min_dist) {
          second_min_dist = dist_1;
        }
      }
    }
  } else {
    extern __shared__ float shared_centroids[];
    for (uint32_t gc = 0; gc < d_clusters_size; gc += tileC) {
      uint32_t tile_count = d_clusters_size - gc;
      if (tile_count > tileC) {
        tile_count = tileC;
      }
      uint32_t tile_elems = tile_count * d_features_size;
      for (uint32_t i = threadIdx.x; i < tile_elems; i += blockDim.x) {
        shared_centroids[i] = centroids[gc * d_features_size + i];
      }
      __syncthreads();

      if (active) {
        for (uint32_t lc = 0; lc < tile_count; lc++) {
          uint32_t c = gc + lc;
          if (c == cluster) {
            continue;
          }
          uint32_t group = groups[c];
          if (group >= d_yy_groups_size) {
            continue;
          }
          float lower_bound = bounds[static_cast<uint64_t>(length) * (1 + group) + sample];
          if (lower_bound >= upper_bound) {
            if (lower_bound < second_min_dist) {
              second_min_dist = lower_bound;
            }
            continue;
          }
          lower_bound += drifts[group] - drifts[doffset + c];
          if (second_min_dist < lower_bound) {
            continue;
          }

          float dist = 0;
          const float *centroid_tile = shared_centroids + lc * d_features_size;
          #pragma unroll 4
          for (uint16_t f = 0; f < d_features_size; ++f) {
            float d = samples[d_samples_size * f + sample] - centroid_tile[f];
            dist += d * d;
          }
          float dist_1 = sqrt(dist);

          if (dist_1 < min_dist) {
            second_min_dist = min_dist;
            min_dist = dist_1;
            nearest = c;
          } else if (dist_1 < second_min_dist) {
            second_min_dist = dist_1;
          }
        }
      }
      __syncthreads();
    }
  }

  if (active) {
    uint32_t nearest_group = groups[nearest];
    uint32_t previous_group = groups[cluster];
    bounds[static_cast<uint64_t>(length) * (1 + nearest_group) + sample] = second_min_dist;

    if (nearest_group != previous_group) {
      uint64_t gindex = static_cast<uint64_t>(length) * (1 + previous_group) + sample;
      float pb = bounds[gindex];
      if (pb > upper_bound) {
        bounds[gindex] = upper_bound;
      }
    }
    bounds[sample] = min_dist;

    if (cluster != nearest) {
      assignments[sample] = nearest;
      atomicAggInc(&d_changed_number);
    }
  }
}


//--------------------------------------------------------------------------------------------------------------------------
template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_calc_average_distance(
    uint32_t offset, uint32_t length, const F *__restrict__ samples,
    const F *__restrict__ centroids, const uint32_t *__restrict__ assignments,
    atomic_float *distance) {
  volatile uint64_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  float dist = 0;
  if (sample < length) {
    sample += offset;
    dist = METRIC<M, F>::distance_t(samples, centroids + assignments[sample] * d_features_size, d_samples_size, sample);
  }
  float sum = warpReduceSum(dist);
  if (threadIdx.x % 32 == 0) {
    atomicAdd(distance, sum);
  }
}

////////////////////------------------------------------------------------------
// Host functions //------------------------------------------------------------
////////////////////------------------------------------------------------------

// Start measure CPU and GPU Power Consumption Using NVIDIA Tegrastats with interavl 100ms
extern "C" void capture_power_metrics_in_interval_start(int iteration) {
    char start_cmd[256];
    snprintf(start_cmd, sizeof(start_cmd), "sudo sh -c 'tegrastats --interval 100 --logfile Power_metric/tegra_log_%d.txt' &", iteration);  // modify file path
    if (system(start_cmd) == -1) {  // tegrastats 시작
        printf("Failed to start tegrastats\n");
    }
}

// Stop measure CPU and GPU Power Consumption
extern "C" void capture_power_metrics_in_interval_end(int iteration) {

  if (system("sudo pkill tegrastats") == -1) {  // tegrastats 종료
      printf("Failed to stop tegrastats\n");
      return;
  }

  char log_file[256];
  snprintf(log_file, sizeof(log_file), "Power_metric/tegra_log_%d.txt", iteration); // modify file path

  // Grant write permission to the log file
  char chmod_cmd[300]; // Allocate large enough buffer
  int chmod_length = snprintf(chmod_cmd, sizeof(chmod_cmd), "sudo chmod 666 %s", log_file);
  
  // Ensure command string fits within buffer
  if (chmod_length >= (int)sizeof(chmod_cmd)) {
      printf("Error: chmod command exceeds buffer size.\n");
      return;
  }

  if (system(chmod_cmd) == -1) {
      printf("Failed to set write permission for log file: %s\n", log_file);
      return;
  }

  // Open the log file in read mode
  FILE *file = fopen(log_file, "r");
  if (file == NULL) {
      printf("Failed to open log file: %s\n", log_file);
      return;
  }
}


static int check_changed(int iter, float tolerance, uint32_t h_samples_size, int32_t verbosity) {
  uint32_t overall_changed = 0;
  CUCH(cudaMemcpyFromSymbol(&overall_changed, d_changed_number, sizeof(overall_changed)), kmcudaMemoryCopyError);  // device -> host

  INFO("Iteration %d: %" PRIu32 " Reassignments\n", iter, overall_changed);

  if (overall_changed <= tolerance * h_samples_size) {
    return -1;
  }
  assert(overall_changed <= h_samples_size);

  uint32_t zero = 0;
  CUCH(cudaMemcpyToSymbolAsync(d_changed_number, &zero, sizeof(zero)), kmcudaMemoryCopyError);  // host -> device
  cudaDeviceSynchronize(); 

  return kmcudaSuccess;
}

static KMCUDAResult prepare_mem(
    uint32_t h_samples_size, uint32_t h_clusters_size, bool resume,
    const std::vector<int> &devs, int verbosity, udevptrs<uint32_t> *ccounts,
    udevptrs<uint32_t> *assignments, udevptrs<uint32_t> *assignments_prev,
    std::vector<uint32_t> *shmem_sizes) {
  uint32_t zero = 0;
  shmem_sizes->clear();
  FOR_EACH_DEVI(
    uint32_t h_shmem_size;
    CUCH(cudaMemcpyFromSymbol(&h_shmem_size, d_shmem_size, sizeof(h_shmem_size)), kmcudaMemoryCopyError);  // device -> host
    shmem_sizes->push_back(h_shmem_size * sizeof(uint32_t));  // add new element in end of vector
    CUCH(cudaMemcpyToSymbolAsync(d_changed_number, &zero, sizeof(zero)), kmcudaMemoryCopyError);
    if (!resume) {
      CUCH(cudaMemsetAsync((*ccounts)[devi].get(), 0, h_clusters_size * sizeof(uint32_t)), kmcudaRuntimeError);
      CUCH(cudaMemsetAsync((*assignments)[devi].get(), 0xff, h_samples_size * sizeof(uint32_t)), kmcudaRuntimeError);
      CUCH(cudaMemsetAsync((*assignments_prev)[devi].get(), 0xff, h_samples_size * sizeof(uint32_t)), kmcudaRuntimeError);
    }
  );
  return kmcudaSuccess;
}

template <typename F>
__global__ void rearrange_for_cpu_batch4(
    const F* __restrict__ samples,
    const uint32_t* __restrict__ mark_cpu,
    F* __restrict__ interleaved_buf,
    uint32_t num_mark_cpu) {
    
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t i = idx % num_mark_cpu; // index in mark_cpu
    uint32_t f = idx / num_mark_cpu; // feature index

    if (f >= d_features_size) return;

    uint32_t sample_idx = mark_cpu[i];
    uint32_t batch_id = i / 4;
    uint32_t offset_in_batch = i % 4;

    // Target layout: [batch_id][f][4] + offset_in_batch
    uint64_t target_idx = static_cast<uint64_t>(batch_id) * d_features_size * 4 + f * 4 + offset_in_batch;
    
    // Source layout (Column-Major): [f][samples_size] + sample_idx
    uint64_t source_idx = static_cast<uint64_t>(f) * d_samples_size + sample_idx;
    
    interleaved_buf[target_idx] = samples[source_idx];
}


extern "C" {
KMCUDAResult kmeans_cuda_setup(
    uint32_t h_samples_size, uint16_t h_features_size, uint32_t h_clusters_size,
    uint32_t h_yy_groups_size, const std::vector<int> &devs, int32_t verbosity) {

  FOR_EACH_DEV(

    printf("\n");

    //DEBUG("-------------------------------------------------------------------------------------------------------------\n");
    CUCH(cudaMemcpyToSymbol(d_samples_size, &h_samples_size, sizeof(h_samples_size)), kmcudaMemoryCopyError);
    //DEBUG("GPU #%" PRIu32 " has %d samples_size\n", dev, h_samples_size);
    CUCH(cudaMemcpyToSymbol(d_features_size, &h_features_size, sizeof(h_features_size)), kmcudaMemoryCopyError);
    //DEBUG("GPU #%" PRIu32 " has %d features_size\n", dev, h_features_size);
    CUCH(cudaMemcpyToSymbol(d_clusters_size, &h_clusters_size, sizeof(h_clusters_size)), kmcudaMemoryCopyError);
    //DEBUG("GPU #%" PRIu32 " has %d clusters_size\n", dev, h_clusters_size);
    CUCH(cudaMemcpyToSymbol(d_yy_groups_size, &h_yy_groups_size, sizeof(h_yy_groups_size)), kmcudaMemoryCopyError);
    //DEBUG("GPU #%" PRIu32 " has %d yy_groups_size\n", dev, h_yy_groups_size);

    cudaDeviceProp props;
    CUCH(cudaGetDeviceProperties(&props, dev), kmcudaRuntimeError);
    int h_shmem_size = static_cast<int>(props.sharedMemPerBlock);
    //DEBUG("GPU #%" PRIu32 " has %d bytes of shared memory per block\n", dev, h_shmem_size); // h_shmem_size 49152
    h_shmem_size /= sizeof(uint32_t);                                                       // transfer unit h_shmem_size = 49152 / 4 = 12288
    CUCH(cudaMemcpyToSymbol(d_shmem_size, &h_shmem_size, sizeof(h_shmem_size)), kmcudaMemoryCopyError);
    //DEBUG("GPU #%" PRIu32 " translated shared memory per block is %d bytes\n", dev, h_shmem_size);
    //DEBUG("-------------------------------------------------------------------------------------------------------------\n");

    printf("\n");

  );

  return kmcudaSuccess;

}

KMCUDAResult kmeans_cuda_plus_plus(
    uint32_t h_samples_size, uint32_t h_features_size, uint32_t cc,
    KMCUDADistanceMetric metric, const std::vector<int> &devs, int fp16x2,
    int verbosity, const udevptrs<float> &samples, udevptrs<float> *centroids,
    udevptrs<float> *dists, float *host_dists, atomic_float *dist_sum) {
  auto plan = distribute(h_samples_size, h_features_size * sizeof(float), devs);
  uint32_t max_len = 0;
  for (auto &p : plan) {
    auto len = std::get<1>(p);
    if (max_len < len) {
      max_len = len;
    }
  }
  udevptrs<atomic_float> dev_dists;
  CUMALLOC(dev_dists, sizeof(atomic_float));
  CUMEMSET_ASYNC(dev_dists, 0, sizeof(atomic_float));
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plan[devi];
    if (length == 0) {
      continue;
    }
    dim3 block(BS_KMPP, 1, 1);
    dim3 grid(upper(length, block.x), 1, 1);
    KERNEL_SWITCH(kmeans_plus_plus, <<<grid, block>>>(
        offset, length, cc,
        reinterpret_cast<const F*>(samples[devi].get()),
        reinterpret_cast<const F*>((*centroids)[devi].get()),
        (*dists)[devi].get(), dev_dists[devi].get()));
  );
  uint32_t dist_offset = 0;
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plan[devi];
    dim3 block(BS_KMPP, 1, 1);
    dim3 grid(upper(length, block.x), 1, 1);
    CUCH(cudaMemcpyAsync(
        host_dists + offset, (*dists)[devi].get(),
        length * sizeof(float), cudaMemcpyDeviceToHost), kmcudaMemoryCopyError);
    dist_offset += grid.x;
  );
  atomic_float sum = 0;
  FOR_EACH_DEVI(
    if (std::get<1>(plan[devi]) == 0) {
      continue;
    }
    atomic_float hdist;
    CUCH(cudaMemcpy(&hdist, dev_dists[devi].get(), sizeof(atomic_float),
                    cudaMemcpyDeviceToHost),
         kmcudaMemoryCopyError);
    sum += hdist;
  );
  *dist_sum = sum;
  return kmcudaSuccess;
}

KMCUDAResult kmeans_cuda_afkmc2_calc_q(
    uint32_t h_samples_size, uint32_t h_features_size, uint32_t firstc,
    KMCUDADistanceMetric metric, const std::vector<int> &devs, int fp16x2,
    int verbosity, const udevptrs<float> &samples, udevptrs<float> *d_q,
    float *h_q) {
  auto plan = distribute(h_samples_size, h_features_size * sizeof(float), devs);
  udevptrs<atomic_float> dev_dists;
  CUMALLOC(dev_dists, sizeof(atomic_float));
  CUMEMSET_ASYNC(dev_dists, 0, sizeof(atomic_float));
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plan[devi];
    if (length == 0) {
      continue;
    }
    dim3 block(BS_AFKMC2_Q, 1, 1);
    dim3 grid(upper(length, block.x), 1, 1);
    int shmem = std::max(
        BS_AFKMC2_Q, static_cast<int>(h_features_size)) * sizeof(float);
    KERNEL_SWITCH(kmeans_afkmc2_calc_q_dists,
                  <<<grid, block, shmem>>>(
        offset, length, firstc,
        reinterpret_cast<const F*>(samples[devi].get()),
        (*d_q)[devi].get(), dev_dists[devi].get()));

  );
  atomic_float dists_sum = 0;
  FOR_EACH_DEVI(
    if (std::get<1>(plan[devi]) == 0) {
      continue;
    }
    atomic_float hdist;
    CUCH(cudaMemcpy(&hdist, dev_dists[devi].get(), sizeof(atomic_float),
                    cudaMemcpyDeviceToHost),
         kmcudaMemoryCopyError);
    dists_sum += hdist;
  );
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plan[devi];
    if (length == 0) {
      continue;
    }
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid(upper(length, block.x), 1, 1);
    kmeans_afkmc2_calc_q<<<grid, block>>>(
        offset, length, dists_sum, (*d_q)[devi].get());
  );
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plan[devi];
    CUCH(cudaMemcpyAsync(h_q + offset, (*d_q)[devi].get() + offset, length * sizeof(float), cudaMemcpyDeviceToHost), kmcudaMemoryCopyError);
    FOR_OTHER_DEVS(
      CUP2P(d_q, offset, length);
    );
  );
  SYNC_ALL_DEVS;
  return kmcudaSuccess;
}

KMCUDAResult kmeans_cuda_afkmc2_random_step(
    uint32_t k, uint32_t m, uint64_t seed, int verbosity, const float *q,
    uint32_t *d_choices, uint32_t *h_choices, float *d_samples, float *h_samples) {
  dim3 block(BS_AFKMC2_R, 1, 1);
  dim3 grid(upper(m, block.x), 1, 1);
  kmeans_afkmc2_random_step<<<grid, block>>>(m, seed, k, q, d_choices, d_samples);
  CUCH(cudaMemcpy(h_choices, d_choices, m * sizeof(uint32_t), cudaMemcpyDeviceToHost), kmcudaMemoryCopyError);
  CUCH(cudaMemcpy(h_samples, d_samples, m * sizeof(float), cudaMemcpyDeviceToHost), kmcudaMemoryCopyError);
  return kmcudaSuccess;
}

KMCUDAResult kmeans_cuda_afkmc2_min_dist(
    uint32_t k, uint32_t m, KMCUDADistanceMetric metric, int fp16x2,
    int32_t verbosity, const float *samples, const uint32_t *choices,
    const float *centroids, float *d_min_dists, float *h_min_dists) {
  if (m > k || m > SHMEM_AFKMC2_MT) {
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid(upper(m, block.x), 1, 1);
    KERNEL_SWITCH(kmeans_afkmc2_min_dist, <<<grid, block>>>(
        m, k, reinterpret_cast<const F*>(samples), choices,
        reinterpret_cast<const F*>(centroids), d_min_dists));
  } else {
    dim3 block(BS_AFKMC2_MDT, 1, 1);
    dim3 grid(upper(k, block.x), 1, 1);
    CUCH(cudaMemsetAsync(d_min_dists, 0xff, m * sizeof(float)), kmcudaRuntimeError);
    KERNEL_SWITCH(kmeans_afkmc2_min_dist_transposed,
        <<<grid, block, m * sizeof(float)>>>(
        m, k, reinterpret_cast<const F*>(samples), choices,
        reinterpret_cast<const F*>(centroids), d_min_dists));
  }
  CUCH(cudaMemcpy(h_min_dists, d_min_dists, m * sizeof(float), cudaMemcpyDeviceToHost), kmcudaMemoryCopyError);
  return kmcudaSuccess;
}

KMCUDAResult kmeans_cuda_lloyd(
    float tolerance, uint32_t h_samples_size, uint32_t h_clusters_size,
    uint16_t h_features_size, KMCUDADistanceMetric metric, bool resume,
    const std::vector<int> &devs, int fp16x2, int32_t verbosity,
    const udevptrs<float> &samples, udevptrs<float> *centroids,
    udevptrs<uint32_t> *ccounts, udevptrs<uint32_t> *assignments_prev,
    udevptrs<uint32_t> *assignments, int *iterations = nullptr) {
  std::vector<uint32_t> shmem_sizes;
  RETERR(prepare_mem(h_samples_size, h_clusters_size, resume, devs, verbosity,
                     ccounts, assignments, assignments_prev, &shmem_sizes));
  auto plans = distribute(h_samples_size, h_features_size * sizeof(float), devs);
  auto planc = distribute(h_clusters_size, h_features_size * sizeof(float), devs);
  if (verbosity > 1) {
    // print_plan("plans", plans);
    // print_plan("planc", planc);
  }
  dim3 sblock(BS_LL_ASS, 1, 1);
  dim3 cblock(BS_LL_CNT, 1, 1);

  cudaEvent_t start_lloyd, stop_lloyd;
  cudaEventCreate(&start_lloyd);
  cudaEventCreate(&stop_lloyd);

  for (int iter = 1; ; iter++) {
    if (!resume || iter > 1) {
      FOR_EACH_DEVI(
        cudaEventRecord(start_lloyd);
        uint32_t offset, length;
        std::tie(offset, length) = plans[devi];
        if (length == 0) {
          continue;
        }
        dim3 sgrid(upper(length, sblock.x), 1, 1);
        int shmem_size = shmem_sizes[devi];
        int64_t ssqrmem = sblock.x * h_features_size * sizeof(float);

        if (shmem_size > ssqrmem && shmem_size - ssqrmem >=
            static_cast<int>((h_features_size + 1) * sizeof(float))) {
          KERNEL_SWITCH(kmeans_assign_lloyd_smallc, <<<sgrid, sblock, shmem_size>>>(
              offset, length,
              reinterpret_cast<const F*>(samples[devi].get()),
              reinterpret_cast<const F*>((*centroids)[devi].get()),
              (*assignments_prev)[devi].get() + offset,
              (*assignments)[devi].get() + offset));
        } else {
          KERNEL_SWITCH(kmeans_assign_lloyd, <<<sgrid, sblock, shmem_size>>>(
              offset, length,
              reinterpret_cast<const F*>(samples[devi].get()),
              reinterpret_cast<const F*>((*centroids)[devi].get()),
              (*assignments_prev)[devi].get() + offset,
              (*assignments)[devi].get() + offset));
        }
        
      );


      FOR_EACH_DEVI(
        uint32_t offset, length;
        std::tie(offset, length) = plans[devi];
        if (length == 0) {
          continue;
        }
        FOR_OTHER_DEVS(
          CUP2P(assignments_prev, offset, length);
          CUP2P(assignments, offset, length);
        );
      );
      int status = check_changed(iter, tolerance, h_samples_size, verbosity);
      //printf("Check Change\n"); // Check for lloyd's K-Means
      if (status < kmcudaSuccess) {
        if (iterations) {
          *iterations = iter+1;
        }
        cudaEventRecord(stop_lloyd);
        cudaEventSynchronize(stop_lloyd);
        float milliseconds_lloyd = 0;
        cudaEventElapsedTime(&milliseconds_lloyd, start_lloyd, stop_lloyd);
        printf("(0) lloyd_assignment execution time: %f ms\n", milliseconds_lloyd);
        cudaEventDestroy(start_lloyd);
        cudaEventDestroy(stop_lloyd);     
        return kmcudaSuccess;
      }
      if (status != kmcudaSuccess) {
        return static_cast<KMCUDAResult>(status);
      }
    }
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = planc[devi];
      if (length == 0) {
        continue;
      }
      dim3 cgrid(upper(length, cblock.x), 1, 1);
      KERNEL_SWITCH(kmeans_adjust, <<<cgrid, cblock, shmem_sizes[devi]>>>(
          offset, length, reinterpret_cast<const F*>(samples[devi].get()),
          (*assignments_prev)[devi].get(), (*assignments)[devi].get(),
          reinterpret_cast<F*>((*centroids)[devi].get()), (*ccounts)[devi].get()));
    );
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = planc[devi];
      if (length == 0) {
        continue;
      }
      FOR_OTHER_DEVS(
        CUP2P(ccounts, offset, length);
        CUP2P(centroids, offset * h_features_size, length * h_features_size);
      );
    );
    cudaEventRecord(stop_lloyd);
    cudaEventSynchronize(stop_lloyd);
    float milliseconds_lloyd = 0;
    cudaEventElapsedTime(&milliseconds_lloyd, start_lloyd, stop_lloyd);
    printf("(0) lloyd_assignment execution time: %f ms\n", milliseconds_lloyd);
  }
  cudaEventDestroy(start_lloyd);
  cudaEventDestroy(stop_lloyd); 
  return kmcudaSuccess;
}

KMCUDAResult kmeans_cuda_yy(
    float tolerance, uint32_t h_yy_groups_size, uint32_t h_samples_size,
    uint32_t h_clusters_size, uint16_t h_features_size, KMCUDADistanceMetric metric,
    const std::vector<int> &devs, int fp16x2, int32_t verbosity,
    const udevptrs<float> &samples, udevptrs<float> *centroids,
    udevptrs<uint32_t> *ccounts, udevptrs<uint32_t> *assignments_prev,
    udevptrs<uint32_t> *assignments, udevptrs<uint32_t> *assignments_yy,
    udevptrs<float> *centroids_yy, //udevptrs<float> *bounds_yy,
    udevptrs<float> *drifts_yy, udevptrs<uint32_t> *passed_yy
    , const float *raw_samples
    ) {

    printf("[KUBISM CONFIG] USE_CUB_PASSED_COMPACTION=%d\n", USE_CUB_PASSED_COMPACTION);
    printf("[KUBISM CONFIG] USE_CUB_MARK_COMPACTION=%d\n", USE_CUB_MARK_COMPACTION);
    printf("[KUBISM CONFIG] KUBISM_WB_HETD_USE_TRANSPOSED_CENTROIDS=%d\n", KUBISM_WB_HETD_USE_TRANSPOSED_CENTROIDS);
    printf("[KUBISM CONFIG] BS_WB_HETD=%d\n", BS_WB_HETD);

    cudaEvent_t start99, stop99;
    cudaEventCreate(&start99);
    cudaEventCreate(&stop99);
    cudaEventRecord(start99); 
      
  //---------------------------------------------------------------------------------
  float* bounds_yy;
  uint32_t yyb_size =  h_samples_size * (h_yy_groups_size + 1) * sizeof(float);
  cudaHostAlloc((void**)&bounds_yy, yyb_size, cudaHostAllocDefault);
  
  uint32_t* assignment_yy;
  cudaHostAlloc((void**)&assignment_yy, h_samples_size * sizeof(uint32_t), cudaHostAllocDefault);
  //---------------------------------------------------------------------------------

  if (h_yy_groups_size == 0 || YINYANG_DRAFT_REASSIGNMENTS <= tolerance) {
    if (verbosity > 0) {
      if (h_yy_groups_size == 0) {
        printf("too few clusters for this yinyang_t => Lloyd\n");
      } else {
        printf("tolerance is too high (>= %.2f) => Lloyd\n", YINYANG_DRAFT_REASSIGNMENTS);
      }
    }
    return kmeans_cuda_lloyd(
        tolerance, h_samples_size, h_clusters_size, h_features_size, metric,
        false, devs, fp16x2, verbosity, samples, centroids, ccounts,
        assignments_prev, assignments);
  }

  INFO("running Lloyd until reassignments drop below %" PRIu32 "\n", (uint32_t)(YINYANG_DRAFT_REASSIGNMENTS * h_samples_size));
  int iter = 0;
  RETERR(kmeans_cuda_lloyd(
      YINYANG_DRAFT_REASSIGNMENTS, h_samples_size, h_clusters_size,
      h_features_size, metric, false, devs, fp16x2, verbosity, samples,
      centroids, ccounts, assignments_prev, assignments,
      &iter));
  if (check_changed(iter, tolerance, h_samples_size, 0) < kmcudaSuccess) {
    return kmcudaSuccess;
  }
  // map each centroid to yinyang group -> assignments_yy
  FOR_EACH_DEV(
    CUCH(cudaMemcpyToSymbol(d_samples_size, &h_clusters_size, sizeof(h_samples_size)), kmcudaMemoryCopyError);    // host -> device
    CUCH(cudaMemcpyToSymbol(d_clusters_size, &h_yy_groups_size, sizeof(h_yy_groups_size)), kmcudaMemoryCopyError);    // host -> device
  );

  {
    udevptrs<float> tmpbufs, tmpbufs2;
    auto max_slength = max_distribute_length(h_samples_size, h_features_size * sizeof(float), devs);
    for (auto &pyy : *passed_yy) {
      // max_slength is guaranteed to be greater than or equal to
      // h_clusters_size + h_yy_groups_size  1024 + 102
      tmpbufs.emplace_back(reinterpret_cast<float*>(pyy.get()) +
          max_slength - h_clusters_size - h_yy_groups_size, true);
      tmpbufs2.emplace_back(tmpbufs.back().get() + h_clusters_size, true);
    }
    RETERR(cuda_transpose(
        h_clusters_size, h_features_size, true, devs, verbosity, centroids));
    RETERR(kmeans_init_centroids(
        kmcudaInitMethodPlusPlus, nullptr, h_clusters_size, h_features_size,
        h_yy_groups_size, metric, 0, devs, -1, fp16x2, verbosity, nullptr,
        *centroids, &tmpbufs, nullptr, centroids_yy),
           INFO("kmeans_init_centroids() failed for yinyang groups: %s\n", cudaGetErrorString(cudaGetLastError())));
    RETERR(kmeans_cuda_lloyd(
        YINYANG_GROUP_TOLERANCE, h_clusters_size, h_yy_groups_size, h_features_size,
        metric, false, devs, fp16x2, verbosity, *centroids, centroids_yy,
        reinterpret_cast<udevptrs<uint32_t> *>(&tmpbufs2),
        reinterpret_cast<udevptrs<uint32_t> *>(&tmpbufs), assignments_yy));
    RETERR(cuda_transpose(
        h_clusters_size, h_features_size, false, devs, verbosity, centroids));

    
      cudaEventRecord(stop99);
      cudaEventSynchronize(stop99);
      
      float time_lloyd_init = 0;
      cudaEventElapsedTime(&time_lloyd_init, start99, stop99);
      printf("(0) Lloyd and centroid group initialization: %f ms\n", time_lloyd_init);
      cudaEventDestroy(start99);
      cudaEventDestroy(stop99); 

  }

    cudaEvent_t start98, stop98;
    cudaEventCreate(&start98);
    cudaEventCreate(&stop98);
    cudaEventRecord(start98); 

  /* h_samples_size, h_clusters_size host to device*/
  FOR_EACH_DEV(
    CUCH(cudaMemcpyToSymbol(d_samples_size, &h_samples_size, sizeof(h_samples_size)), kmcudaMemoryCopyError); // host -> device
    CUCH(cudaMemcpyToSymbol(d_clusters_size, &h_clusters_size, sizeof(h_clusters_size)), kmcudaMemoryCopyError); // host -> device
  );

  std::vector<uint32_t> shmem_sizes;
  RETERR(prepare_mem(h_samples_size, h_clusters_size, true, devs, verbosity, ccounts, 
  assignments, assignments_prev, &shmem_sizes));

#if USE_CUB_PASSED_COMPACTION
  udevptrs<uint8_t> passed_flags;
  udevptrs<uint32_t> passed_counts;
  udevptrs<uint8_t> passed_temp;
  size_t passed_temp_bytes = 0;
  {
    auto max_slength = max_distribute_length(h_samples_size, h_features_size * sizeof(float), devs);
    cub::CountingInputIterator<uint32_t> counting(0);
    cub::DeviceSelect::Flagged(nullptr, passed_temp_bytes, counting,
                               static_cast<uint8_t *>(nullptr),
                               static_cast<uint32_t *>(nullptr),
                               static_cast<uint32_t *>(nullptr),
                               max_slength);
    CUMALLOC(passed_flags, max_slength);
    CUMALLOC(passed_counts, 1);
    CUMALLOC(passed_temp, passed_temp_bytes);
  }
#endif

#if USE_CUB_MARK_COMPACTION
  udevptrs<uint8_t> mark_cpu_flags;
  udevptrs<uint8_t> mark_gpu_flags;
  udevptrs<uint32_t> mark_cpu_counts;
  udevptrs<uint32_t> mark_gpu_counts;
  udevptrs<uint8_t> mark_temp;
  size_t mark_temp_bytes = 0;
  {
    auto max_slength = max_distribute_length(h_samples_size, h_features_size * sizeof(float), devs);
    cub::DeviceSelect::Flagged(nullptr, mark_temp_bytes,
                               static_cast<const uint32_t *>(nullptr),
                               static_cast<const uint8_t *>(nullptr),
                               static_cast<uint32_t *>(nullptr),
                               static_cast<uint32_t *>(nullptr),
                               max_slength);
    CUMALLOC(mark_cpu_flags, max_slength);
    CUMALLOC(mark_gpu_flags, max_slength);
    CUMALLOC(mark_cpu_counts, 1);
    CUMALLOC(mark_gpu_counts, 1);
    CUMALLOC(mark_temp, mark_temp_bytes);
  }
#endif

#if KUBISM_WB_HETD_USE_TRANSPOSED_CENTROIDS
  udevptrs<float> centroids_gpu_soa;
  CUMALLOC(centroids_gpu_soa,
           static_cast<uint64_t>(h_clusters_size) * h_features_size);
#endif

  
  dim3 siblock(BS_YY_INI, 1, 1);
  dim3 sgblock(BS_YY_GFL, 1, 1);
  dim3 slblock(BS_YY_LFL, 1, 1);
  dim3 cblock(BS_LL_CNT, 1, 1);
  dim3 gblock(BLOCK_SIZE, 1, 1);

  dim3 wb_hetd_block(BS_WB_HETD, 1, 1);
  auto plans = distribute(h_samples_size, h_features_size * sizeof(float), devs); // sample data 
  auto planc = distribute(h_clusters_size, h_features_size * sizeof(float), devs); // centroid data
  auto plang = distribute(h_yy_groups_size, h_features_size * sizeof(float), devs); // centroid group data 

  if (verbosity > 1) {
    // print_plan("plans", plans);
    // print_plan("planc", planc);
    // print_plan("plang", plang);
  }

  bool refresh = true;
 
  uint32_t h_passed_number = 0; // initialization

  float     *h_samples        = (float*)malloc(h_samples_size * h_features_size * sizeof(float)); 

  FOR_EACH_DEVI(
    cudaMemcpy(h_samples, (samples)[devi].get(), h_samples_size * h_features_size * sizeof(float), cudaMemcpyDeviceToHost);
  );


      cudaEventRecord(stop98);
      cudaEventSynchronize(stop98);
      float time_memcopy_init = 0;
      cudaEventElapsedTime(&time_memcopy_init, start98, stop98);
      printf("(0)  h_samples_size, h_clusters_size memcopy, initialization, hsample_memcopy: %f ms\n", time_memcopy_init);
      cudaEventDestroy(start98);
      cudaEventDestroy(stop98); 


  // Pre-allocate reusable CUDA events for measurement
  cudaEvent_t event_start, event_stop;
  cudaEventCreate(&event_start);
  cudaEventCreate(&event_stop);
  // Separate events for per-iteration total timing
  cudaEvent_t iter_start, iter_stop;
  cudaEventCreate(&iter_start);
  cudaEventCreate(&iter_stop);

  // Reusable pinned memory (single GPU, allocate once)
  int32_t *mark_threads_yy;
  cudaHostAlloc(&mark_threads_yy, (h_samples_size) * (h_clusters_size) * sizeof(int32_t), cudaHostAllocMapped);




  float *second_min_dists_yy;
  cudaHostAlloc(&second_min_dists_yy, static_cast<size_t>(h_samples_size) * sizeof(float), cudaHostAllocMapped);

  uint32_t *calculate_sum;
  cudaHostAlloc((void**)&calculate_sum, sizeof(uint32_t), cudaHostAllocMapped);

  float current_skip_ratio = 0.0f;




  uint32_t *mark_cpu_yy;
  cudaHostAlloc((void**)&mark_cpu_yy, static_cast<uint32_t>(h_samples_size) * sizeof(uint32_t), cudaHostAllocMapped);

  uint32_t *mark_gpu_yy;
  cudaHostAlloc((void**)&mark_gpu_yy, static_cast<uint32_t>(h_samples_size) * sizeof(uint32_t), cudaHostAllocMapped);

  uint32_t *mark_cpu_number;
  cudaHostAlloc(&mark_cpu_number, sizeof(uint32_t), cudaHostAllocMapped);

  uint32_t *mark_gpu_number;
  cudaHostAlloc(&mark_gpu_number, sizeof(uint32_t), cudaHostAllocMapped);

  float     *h_centroids      = (float*)malloc(h_clusters_size * h_features_size * sizeof(float));
  uint32_t  *h_assignments_yy = (uint32_t*)malloc(h_clusters_size * sizeof(uint32_t));
  float     *h_drifts_yy      = (float*)malloc((h_clusters_size) * (h_features_size + 1) * sizeof(float));
  uint32_t  *h_passed_yy      = (uint32_t*)malloc(h_samples_size * sizeof(uint32_t));
  float *h_interleaved_samples;
  float *d_interleaved_samples;
  cudaHostAlloc(&h_interleaved_samples, h_samples_size * h_features_size * sizeof(float), cudaHostAllocMapped);
  cudaHostGetDevicePointer(&d_interleaved_samples, h_interleaved_samples, 0);

  INFO("Pre-faulting pinned memory...\n");
  auto prefault_start = std::chrono::high_resolution_clock::now();
  memset(mark_threads_yy, 0, (h_samples_size) * static_cast<int32_t>(h_clusters_size) * sizeof(int32_t));
  memset(second_min_dists_yy, 0, static_cast<uint32_t>(h_samples_size) * sizeof(float));
  auto prefault_end = std::chrono::high_resolution_clock::now();
  float prefault_time = std::chrono::duration<float, std::milli>(prefault_end - prefault_start).count();
  printf("Pre-faulting done. Execution time: %f ms\n", prefault_time);

    //------------------------------------------------------iter-------------------------------------------------------------

  for (; ; iter++) {

    if (!refresh) { // refresh: decine whether initialize bounds -> do not initialize
      cudaEventRecord(event_start);
      int status = check_changed(iter-1, tolerance, h_samples_size, verbosity);
      if (status < kmcudaSuccess) {
        return kmcudaSuccess;
      }
      if (status != kmcudaSuccess) {
        return static_cast<KMCUDAResult>(status);
      }
      FOR_EACH_DEV(
        uint32_t local_passed;
        CUCH(cudaMemcpyFromSymbol(&local_passed, d_passed_number, sizeof(h_passed_number)), kmcudaMemoryCopyError); // device -> host
        h_passed_number += local_passed;
      );
      if (1.f - (h_passed_number + 0.f) / h_samples_size < YINYANG_REFRESH_EPSILON) {
        refresh = true;
        float cnt = 1.f - (h_passed_number + 0.f) / h_samples_size < YINYANG_REFRESH_EPSILON;
        // printf("start initialize! cnt = %f\n", cnt);
       
      }
      h_passed_number = 0;
      cudaEventRecord(event_stop);
      cudaEventSynchronize(event_stop);
      
      float time_check_init = 0;
      cudaEventElapsedTime(&time_check_init, event_start, event_stop);
      printf("(1) Check initialization: %f ms\n", time_check_init);
    }
//------------------------------------------------------------------------------------yinyang 1---------------------------------------------------------------------------

    if (refresh) {
          // initialize
      printf("\n");
      INFO("Refreshing Yinyang Bounds, Call Bound Initialization Kernel\n");
      printf("\n");

      FOR_EACH_DEVI(

        cudaEventRecord(event_start); 
        uint32_t offset, length; // offset: start of data, length: size of data 
        std::tie(offset, length) = plans[devi];

        if (length == 0) {  // if there is no daa that device will process, 
          continue;
        }
            

        dim3 sigrid(upper(length, siblock.x), 1, 1);
        KERNEL_SWITCH(kmeans_yy_init, <<<sigrid, siblock, shmem_sizes[devi]>>>(
              offset, length,
              reinterpret_cast<const F*>(samples[devi].get()),
              reinterpret_cast<const F*>((*centroids)[devi].get()),
              (*assignments)[devi].get() + offset,
              (*assignments_yy)[devi].get(), 
              bounds_yy));
        cudaEventRecord(event_stop);
        cudaEventSynchronize(event_stop);
        
        float time_kernel_init = 0;
        cudaEventElapsedTime(&time_kernel_init, event_start, event_stop);
        printf("(1) Initialization kernel execution time: %f ms\n", time_kernel_init);
        
      );

      refresh = false;
   
   
    }
    
//----------------------------------------------------------------------------yinyang 2-----------------------------------------------------------------------------------
    printf("\n[Current Iteration = %d]\n", iter);

    cudaEventRecord(iter_start);

    // [New] Spawn background thread to initialize host memory asynchronously before it is actually needed!
    std::thread memset_thread([&]() {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        memset(mark_threads_yy, 0, (h_samples_size) * static_cast<int32_t>(h_clusters_size) * sizeof(int32_t));

        memset(second_min_dists_yy, 0, static_cast<uint32_t>(h_samples_size) * sizeof(float));
        *calculate_sum = 0;


        auto cpu_end = std::chrono::high_resolution_clock::now();
        float memset_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
        printf("(0-1) Async CPU Memset execution time: %f ms\n", memset_time);
    });

    CUMEMCPY_D2D_ASYNC(*drifts_yy, 0, *centroids, 0, h_clusters_size * h_features_size);  // asynchronously copy data from centroids to drifts
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = planc[devi];   // tuple(offset, length)
      if (length == 0) {
        continue;
      }

      cudaEventRecord(event_start);

      dim3 cgrid(upper(length, cblock.x), 1, 1);
      KERNEL_SWITCH(kmeans_adjust, <<<cgrid, cblock, shmem_sizes[devi]>>>(
          offset, length, reinterpret_cast<const F*>(samples[devi].get()),
          (*assignments_prev)[devi].get(), (*assignments)[devi].get() + offset,
          reinterpret_cast<F*>((*centroids)[devi].get()), (*ccounts)[devi].get()));

      cudaEventRecord(event_stop);
      cudaEventSynchronize(event_stop);

      float time_cluster_adjustment = 0;

      cudaEventElapsedTime(&time_cluster_adjustment, event_start, event_stop);
      printf("(2) cluster adjustment: %f ms\n", time_cluster_adjustment); 
      
    );
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = planc[devi];
      if (length == 0) {
        continue;
      }
      FOR_OTHER_DEVS( // if need. copy to other device, little overhead in jetson system
        CUP2P(ccounts, offset, length);
        CUP2P(centroids, offset * h_features_size, length * h_features_size);
      );
    );
//----------------------------------------------------------------------------yinyang 3------------------------------------------------------------------------------------
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = planc[devi];
      if (length == 0) {
        continue;
      }


      cudaEventRecord(event_start);  

      dim3 cgrid(upper(length, cblock.x), 1, 1);
      KERNEL_SWITCH(kmeans_yy_calc_drifts, <<<cgrid, cblock>>>(
          offset, length, reinterpret_cast<const F*>((*centroids)[devi].get()),
          reinterpret_cast<F*>((*drifts_yy)[devi].get())));
      cudaEventRecord(event_stop);
      cudaEventSynchronize(event_stop);

      float time_drift_calc = 0;
      cudaEventElapsedTime(&time_drift_calc, event_start, event_stop);
     printf("(3) Drift Calculation kernel execution time: %f ms\n", time_drift_calc);
      
    );
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = planc[devi];
      if (length == 0) {
        continue;
      }
      FOR_OTHER_DEVS(
        CUP2P(drifts_yy, h_clusters_size * h_features_size + offset, length);
      );
    );
//---------------------------------------------------------------------------------yinyang 4--------------------------------------------------------------------------------
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = plang[devi];
      if (length == 0) {
        continue;
      }
      
      cudaEventRecord(event_start);
      dim3 ggrid(upper(length, gblock.x), 1, 1);
      kmeans_yy_find_group_max_drifts<<<ggrid, gblock, shmem_sizes[devi]>>>(
          offset, length, (*assignments_yy)[devi].get(),
          (*drifts_yy)[devi].get());
      cudaEventRecord(event_stop);
      cudaEventSynchronize(event_stop);
      float time_group_max_drifts = 0;
      cudaEventElapsedTime(&time_group_max_drifts, event_start, event_stop);
      printf("(4) Group Max Drifts kernel execution time: %f ms\n", time_group_max_drifts);
      
    );
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = plang[devi];
      if (length == 0) {
        continue;
      }
      FOR_OTHER_DEVS(
        CUP2P(drifts_yy, offset, length);
      );
    );

    FOR_EACH_DEV(
      CUCH(cudaMemcpyToSymbolAsync(d_passed_number, &h_passed_number, sizeof(h_passed_number)), kmcudaMemoryCopyError);  // host -> device
      );
//--------------------------------------------------------------------------------yinyang 5-------------------------------------------------------------------------------
   
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = plans[devi];
      if (length == 0) {
        continue;
      }

#if USE_CUB_PASSED_COMPACTION
      CUCH(cudaMemsetAsync(passed_flags[devi].get(), 0, length * sizeof(uint8_t)), kmcudaRuntimeError);
#endif
      
      cudaEventRecord(event_start);
      dim3 sggrid(upper(length, sgblock.x), 1, 1);
      KERNEL_SWITCH(kmeans_yy_global_filter, <<<sggrid, sgblock>>>(
          offset, length,
          reinterpret_cast<const F*>(samples[devi].get()),
          reinterpret_cast<const F*>((*centroids)[devi].get()),
          (*assignments_yy)[devi].get(), 
          (*drifts_yy)[devi].get(),
          //drifts_yyy,
          (*assignments)[devi].get() + offset, 
          (*assignments_prev)[devi].get() + offset,
          bounds_yy,
          //(*bounds_yy)[devi].get(),
          (*passed_yy)[devi].get()
    #if USE_CUB_PASSED_COMPACTION
          , passed_flags[devi].get()
    #endif
          ));
      cudaEventRecord(event_stop);
      cudaEventSynchronize(event_stop);

      float time_global_filter = 0;
      cudaEventElapsedTime(&time_global_filter, event_start, event_stop);

    #if USE_CUB_PASSED_COMPACTION
      {
        cudaEvent_t cub_start, cub_stop;
        cudaEventCreate(&cub_start);
        cudaEventCreate(&cub_stop);
        cudaEventRecord(cub_start);

        cub::CountingInputIterator<uint32_t> counting(0);
        cub::DeviceSelect::Flagged(passed_temp[devi].get(), passed_temp_bytes, counting,
                   passed_flags[devi].get(),
                   (*passed_yy)[devi].get(),
                   passed_counts[devi].get(),
                   length);

        cudaEventRecord(cub_stop);
        cudaEventSynchronize(cub_stop);
        float cub_time = 0;
        cudaEventElapsedTime(&cub_time, cub_start, cub_stop);
        printf("(5-1) CUB Passed Compaction time: %f ms\n", cub_time);
        
        cudaEventDestroy(cub_start);
        cudaEventDestroy(cub_stop);
      }
    #endif

      uint32_t global_to_local; // the number of data points which filtered at group filter
    #if USE_CUB_PASSED_COMPACTION
      CUCH(cudaMemcpy(&global_to_local, passed_counts[devi].get(), sizeof(uint32_t), cudaMemcpyDeviceToHost), kmcudaMemoryCopyError);
      CUCH(cudaMemcpyToSymbolAsync(d_passed_number, &global_to_local, sizeof(uint32_t)), kmcudaMemoryCopyError);
    #else
      CUCH(cudaMemcpyFromSymbol(&global_to_local, d_passed_number, sizeof(uint32_t)), kmcudaMemoryCopyError);
    #endif
      printf("(5) Iteration %d Global Filter to Local Filter: %u\n", iter, global_to_local);
      printf("(5) global Filter: %f ms\n", time_global_filter);     

//-----------------------------------------------------Start Power Measure---------------------------------------------------------------------------------------------
      // std::cout << "Start capturing power metrics..." << std::endl;
      // capture_power_metrics_in_interval_start(iter);
//------------------------------------------------------Memory Allocation-------------------------------------------------------------

  // Ensure the background memory initialization is complete before using the arrays
  memset_thread.join();
  uint32_t h_changed_number_cpu = 0;
  
//---------------------------------------------------------Baseline---------------------------------------------------------------- 
#ifdef FUTURISM_EXECUTION_MODE_BASELINE
        cudaEventRecord(event_start);
        dim3 baseline_block(512, 1, 1);
        dim3 baseline_grid(upper(length, baseline_block.x), 1, 1);
        KERNEL_SWITCH(kmeans_yinyang_local_filter, <<<baseline_grid,  baseline_block, shmem_sizes[devi]>>>(
            length, reinterpret_cast<const float*>(samples[devi].get()), (*passed_yy)[devi].get(),
            reinterpret_cast<const float*>((*centroids)[devi].get()),
            (*assignments_yy)[devi].get(), 
            (*drifts_yy)[devi].get(),    // assignments_yy equal to groups
            (*assignments)[devi].get() + offset, 
            bounds_yy,
            second_min_dists_yy
            ));
        cudaEventRecord(event_stop);
        cudaEventSynchronize(event_stop);
        float time_baseline_filter = 0;
        cudaEventElapsedTime(&time_baseline_filter, event_start, event_stop);
        printf("(6) Local Filter Baseline kernel execution time: %f ms\n", time_baseline_filter);
#else
    //-----------------------------------------------Clustering Reordering----------------------------------------------------------------
    {
      cudaEventRecord(event_start);

      dim3 slgrid(upper(length, slblock.x), 1, 1);
      KERNEL_SWITCH(kubism_clustering_reordering, <<<slgrid, slblock, shmem_sizes[devi]>>>(
          length, (*passed_yy)[devi].get(),
          (*assignments_yy)[devi].get(), 
          (*drifts_yy)[devi].get(),    // assignments_yy equal to groups
          (*assignments)[devi].get() + offset, 
          bounds_yy,
          second_min_dists_yy,
          calculate_sum));
      cudaEventRecord(event_stop);
      cudaEventSynchronize(event_stop);
      float time_reordering = 0;
      cudaEventElapsedTime(&time_reordering, event_start, event_stop);
      printf("(6) Early Skip Ratio Marking kernel execution time(1_1): %f ms\n", time_reordering);
//------------------------------------------------Calculate Skip Ratio--------------------------------------------     
      auto cpu_start_sr = std::chrono::high_resolution_clock::now();

      current_skip_ratio = 1.0f - (static_cast<float>(*calculate_sum) / static_cast<float>(global_to_local * h_clusters_size));

      auto cpu_end_sr = std::chrono::high_resolution_clock::now();
      float time_calc_skip_ratio = std::chrono::duration<float, std::milli>(cpu_end_sr - cpu_start_sr).count();
      printf("(7) Calculation Skip Ratio execution time: %f ms\n", time_calc_skip_ratio);
      // printf("Current iteration Skip Ratio = %0.5f\n", current_skip_ratio);

      
    }
      
//----------------------------------------Warp Balancing + HETD-----------------------------------------------------

      cudaEvent_t start6, stop6;
      cudaEventCreate(&start6);
      cudaEventCreate(&stop6);
      cudaEventRecord(start6);

      dim3 slgrid(upper(length, slblock.x), 1, 1);
      KERNEL_SWITCH(kubism_warp_balancing, <<<slgrid, slblock, shmem_sizes[devi]>>>(
          length, (*passed_yy)[devi].get(),
          (*assignments_yy)[devi].get(), 
          (*drifts_yy)[devi].get(),    // assignments_yy equal to groups
          (*assignments)[devi].get() + offset, 
          bounds_yy,
          second_min_dists_yy,
          mark_threads_yy));
      cudaDeviceSynchronize();

      cudaEventRecord(stop6);
      cudaEventSynchronize(stop6);
      float time_warp_balancing = 0;
      cudaEventElapsedTime(&time_warp_balancing, start6, stop6);
      printf("(6) Local Filter Marking kernel execution time: %f ms\n", time_warp_balancing);
  
      cudaEventDestroy(start6);
      cudaEventDestroy(stop6); 
//----------------------------------------------------------------------------------------------------------------------

      cudaEventRecord(event_start);

    //------------------------------------------------HETD (CPU & GPU Task Distribution)-------------------------------------------------------------------------------------------- 

      memset(mark_cpu_yy, 0, static_cast<uint32_t>(h_samples_size) * sizeof(uint32_t));
      memset(mark_gpu_yy, 0, static_cast<uint32_t>(h_samples_size) * sizeof(uint32_t));

      memset(mark_cpu_number, 0, sizeof(uint32_t)); // Set to 0
      memset(mark_gpu_number, 0, sizeof(uint32_t)); // Set to 0

    #if USE_CUB_MARK_COMPACTION
      CUCH(cudaMemsetAsync(mark_cpu_flags[devi].get(), 0, global_to_local * sizeof(uint8_t)), kmcudaRuntimeError);
      CUCH(cudaMemsetAsync(mark_gpu_flags[devi].get(), 0, global_to_local * sizeof(uint8_t)), kmcudaRuntimeError);
    #endif
      
      KERNEL_SWITCH(kubism_CPU_GPU_task_partition, <<<slgrid, slblock>>>((*passed_yy)[devi].get(), 
                                                                            mark_cpu_yy, mark_gpu_yy, current_skip_ratio,
                                mark_cpu_number, mark_gpu_number
      #if USE_CUB_MARK_COMPACTION
        , mark_cpu_flags[devi].get(), mark_gpu_flags[devi].get()
      #endif
        ));
    #if USE_CUB_MARK_COMPACTION
      {
        cub::DeviceSelect::Flagged(mark_temp[devi].get(), mark_temp_bytes,
                   (*passed_yy)[devi].get(),
                   mark_cpu_flags[devi].get(),
                   mark_cpu_yy,
                   mark_cpu_counts[devi].get(),
                   global_to_local);
        cub::DeviceSelect::Flagged(mark_temp[devi].get(), mark_temp_bytes,
                   (*passed_yy)[devi].get(),
                   mark_gpu_flags[devi].get(),
                   mark_gpu_yy,
                   mark_gpu_counts[devi].get(),
                   global_to_local);
      }
      uint32_t mark_cpu_count = 0;
      uint32_t mark_gpu_count = 0;
      CUCH(cudaMemcpy(&mark_cpu_count, mark_cpu_counts[devi].get(), sizeof(uint32_t), cudaMemcpyDeviceToHost), kmcudaMemoryCopyError);
      CUCH(cudaMemcpy(&mark_gpu_count, mark_gpu_counts[devi].get(), sizeof(uint32_t), cudaMemcpyDeviceToHost), kmcudaMemoryCopyError);
      *mark_cpu_number = mark_cpu_count;
      *mark_gpu_number = mark_gpu_count;
    #endif
      cudaDeviceSynchronize();

      cudaEventRecord(event_stop);
      cudaEventSynchronize(event_stop);
      float time_cpu_gpu_partition = 0;
      cudaEventElapsedTime(&time_cpu_gpu_partition, event_start, event_stop);
      
      printf("(8) CPU-GPU Data Point Partition kernel execution time: %f ms\n", time_cpu_gpu_partition); 


      printf("CPU Threads = %u GPU Threads = %u Thread ratio = %.2f\n", *mark_cpu_number, *mark_gpu_number, (static_cast<float>(*mark_gpu_number)/static_cast<float>(*mark_cpu_number)));

//-------------------------------------------------------HETD (Device To Host)------------------------------------------------------------------------------------ 
      
      cudaEvent_t start_d2h, stop_d2h;
      cudaEventCreate(&start_d2h);
      cudaEventCreate(&stop_d2h);
      cudaEventRecord(start_d2h);
      
      cudaMemcpy(assignment_yy, (*assignments)[devi].get(), h_samples_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_centroids, (*centroids)[devi].get(), h_clusters_size * h_features_size * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_assignments_yy, (*assignments_yy)[devi].get(), h_clusters_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_drifts_yy, (*drifts_yy)[devi].get(), (h_clusters_size) * (h_features_size + 1) * sizeof(float), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();

      cudaEventRecord(stop_d2h);
      cudaEventSynchronize(stop_d2h);
      float milliseconds_d2h = 0;
      cudaEventElapsedTime(&milliseconds_d2h, start_d2h, stop_d2h);
      printf("(9-3-3) Device to Host Data Transfer Time: %f ms\n", milliseconds_d2h);
      cudaEventDestroy(start_d2h);
      cudaEventDestroy(stop_d2h);

//-----------------------------------------------------------HETD (CPU GPU HETD parallel)---------------------------------------------------------------------------------
    #if KUBISM_WB_HETD_USE_TRANSPOSED_CENTROIDS
      cudaEvent_t soa_tp_start, soa_tp_stop;
      cudaEventCreate(&soa_tp_start);
      cudaEventCreate(&soa_tp_stop);
      cudaEventRecord(soa_tp_start); 
    #endif   
    
    #if KUBISM_WB_HETD_USE_TRANSPOSED_CENTROIDS
      CUMEMCPY_D2D(centroids_gpu_soa, 0, *centroids, 0,
                  h_clusters_size * h_features_size);
      RETERR(cuda_transpose(
          h_clusters_size, h_features_size, true, devs, verbosity,
          &centroids_gpu_soa));
    #endif

    

    #if KUBISM_WB_HETD_USE_TRANSPOSED_CENTROIDS
        const float *gpu_centroids =
          reinterpret_cast<const float*>(centroids_gpu_soa[devi].get());
    #else
        const float *gpu_centroids =
          reinterpret_cast<const float*>((*centroids)[devi].get());
    #endif
    
    #if KUBISM_WB_HETD_USE_TRANSPOSED_CENTROIDS
    cudaEventRecord(soa_tp_stop);
    cudaEventSynchronize(soa_tp_stop);

    float milliseconds_tp = 0;
    cudaEventElapsedTime(&milliseconds_tp, soa_tp_start, soa_tp_stop);
    printf("(9-3-4) transpose overhead: %f ms\n", milliseconds_tp);
    cudaEventDestroy(soa_tp_start);
    cudaEventDestroy(soa_tp_stop);
    #endif

#if USE_WB_HETD_BATCH4
      cudaEvent_t rearrange_start, rearrange_stop;
      cudaEventCreate(&rearrange_start);
      cudaEventCreate(&rearrange_stop);
      cudaEventRecord(rearrange_start);

      dim3 block(BLOCK_SIZE, 1, 1);
      dim3 grid(upper((*mark_cpu_number) * (uint32_t)h_features_size, (uint32_t)block.x), 1, 1);
      rearrange_for_cpu_batch4<float><<<grid, block>>>(
          reinterpret_cast<const float*>(samples[devi].get()), 
          mark_cpu_yy, d_interleaved_samples, *mark_cpu_number);

      cudaEventRecord(rearrange_stop);
      cudaEventSynchronize(rearrange_stop);
      float milliseconds_rearrange = 0;
      cudaEventElapsedTime(&milliseconds_rearrange, rearrange_start, rearrange_stop);
      printf("(9-3-5) rearrange overhead: %f ms\n", milliseconds_rearrange);
      cudaEventDestroy(rearrange_start);
      cudaEventDestroy(rearrange_stop);
#endif



      cudaEvent_t kernel_start;
      cudaEvent_t kernel_stop;
      cudaEvent_t  sync_start;

      cudaEventCreate(&kernel_start);
      cudaEventCreate(&kernel_stop);
      cudaEventCreate(&sync_start);


      cudaEventRecord(kernel_start);

      const int sharedMemSize = 48 * 1024;
      dim3 wb_hetd_grid(upper(*mark_gpu_number, wb_hetd_block.x), 1, 1);

      CUCH(cudaFuncSetAttribute(
          kubism_GPU_WB_HETD<kmcudaDistanceMetricL2, float>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          sharedMemSize), kmcudaRuntimeError);

      kubism_GPU_WB_HETD<kmcudaDistanceMetricL2, float><<<wb_hetd_grid, wb_hetd_block,
        sharedMemSize>>>(
          length,
          reinterpret_cast<const float*>(samples[devi].get()),
          mark_gpu_yy,
          gpu_centroids,
          (*assignments_yy)[devi].get(), // assignments_yy equal to groups
          assignment_yy,
          bounds_yy,
          second_min_dists_yy, mark_gpu_number,
          mark_threads_yy
          );

      cudaEventRecord(kernel_stop);


#if USE_WB_HETD_BATCH4
      kubism_CPU_WB_HETD_batch4<float>(h_samples_size, h_features_size,
                                      h_interleaved_samples, mark_cpu_yy, h_centroids,
                                      h_assignments_yy, assignment_yy, bounds_yy,
                                      second_min_dists_yy, mark_cpu_number,
                                      h_changed_number_cpu, mark_threads_yy);
#else
     kubism_CPU_WB_HETD<float>(h_samples_size, h_clusters_size, h_features_size, h_yy_groups_size,
#if USE_RAW_SAMPLES
                              raw_samples,
#else
                              h_samples,
#endif
                              mark_cpu_yy, h_centroids, h_assignments_yy, h_drifts_yy, 
                              assignment_yy, bounds_yy, second_min_dists_yy, 
                              mark_cpu_number, h_changed_number_cpu,
                              mark_threads_yy);
#endif
      
      cudaDeviceSynchronize();
      cudaEventRecord(sync_start);
      cudaEventSynchronize(sync_start);

      float kernel_time = 0;
      cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);
      printf("(9-3-4) RWD kernel execution time: %f ms\n", kernel_time);

      float sync_time1 = 0;
      cudaEventElapsedTime(&sync_time1, kernel_start, sync_start);
      printf("(9-3-4) kernel start ~ sync start: %f ms\n", sync_time1);

      cudaEventDestroy(kernel_start);
      cudaEventDestroy(kernel_stop);
      cudaEventDestroy(sync_start);

//---------------------------------------------------HETD (Host To Device)---------------------------------------------------------------------------------------  
      cudaEvent_t start_h2d, stop_h2d;
      cudaEventCreate(&start_h2d);
      cudaEventCreate(&stop_h2d);
      cudaEventRecord(start_h2d);
  

      cudaMemcpy((*assignments)[devi].get(), assignment_yy, h_samples_size * sizeof(uint32_t), cudaMemcpyHostToDevice);

      cudaDeviceSynchronize();
      cudaEventRecord(stop_h2d);
      cudaEventSynchronize(stop_h2d);

      float milliseconds_h2d = 0;
      cudaEventElapsedTime(&milliseconds_h2d, start_h2d, stop_h2d);
      printf("(9-3-5) Host To Device Data Transfer Time: %f ms\n", milliseconds_h2d);

      cudaEventDestroy(start_h2d);
      cudaEventDestroy(stop_h2d);
#endif
//----------------------------------------------------------------------------------------------------------------------------------------------------------------  
      /* Retrieve d_changed_number from GPU */
      uint32_t h_changed_number_gpu;
      CUCH(cudaMemcpyFromSymbol(&h_changed_number_gpu, d_changed_number, sizeof(uint32_t), 0, cudaMemcpyDeviceToHost), kmcudaMemoryCopyError); // device -> host
      cudaDeviceSynchronize();
      /* Combine CPU and GPU changed numbers */
      uint32_t total_changed_number = h_changed_number_cpu + h_changed_number_gpu;
      printf("(9-3-6) Iteration %d Total Reassignments (CPU + GPU): %u\n", iter, total_changed_number );
      CUCH(cudaMemcpyToSymbol(d_changed_number, &total_changed_number, sizeof(uint32_t), 0, cudaMemcpyHostToDevice), kmcudaMemoryCopyError); // reflect next iteration
//---------------------------------------------------------------------------------------------------------------------------------------------------------------- 
      /* Current data to Prev data */

      printf("current_skip_ratio = %0.3f \n", current_skip_ratio);



  // (reusable pinned memory freed after loop)

//-----------------------------------------------------Stop Power Measure---------------------------------------------------------------------------------------------
  // std::cout << "End capturing power metrics..." << std::endl;   
  // capture_power_metrics_in_interval_end(iter);

  );

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = plans[devi];
      if (length == 0) {
        continue;
      }
      FOR_OTHER_DEVS(
        CUP2P(assignments_prev, offset, length);
        CUP2P(assignments, offset, length);
      );
    );

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    cudaEventRecord(iter_stop);
    cudaEventSynchronize(iter_stop);
    float milliseconds0 = 0;
    cudaEventElapsedTime(&milliseconds0, iter_start, iter_stop);
    printf("Iteration %d time: %f ms\n",iter, milliseconds0);



  }
  //------------------------------------------------------iter-------------------------------------------------------------
  cudaFreeHost(second_min_dists_yy);
  cudaFreeHost(mark_threads_yy);
  cudaFreeHost(calculate_sum);
  cudaFreeHost(mark_cpu_yy);
  cudaFreeHost(mark_gpu_yy);
  cudaFreeHost(mark_cpu_number);
  cudaFreeHost(mark_gpu_number);
  cudaEventDestroy(event_start);
  cudaEventDestroy(event_stop);
  cudaEventDestroy(iter_start);
  cudaEventDestroy(iter_stop);
  cudaFreeHost(bounds_yy);
  cudaFreeHost(assignment_yy);
  free(h_centroids);
  free(h_assignments_yy);
  free(h_drifts_yy);
  free(h_passed_yy);
  free(h_samples);
  return kmcudaSuccess;
}
//------------------------------------------------------kmeans_cuda_yy------------------------------------------------------------

KMCUDAResult kmeans_cuda_calc_average_distance(
    uint32_t h_samples_size, uint16_t h_features_size,
    KMCUDADistanceMetric metric, const std::vector<int> &devs, int fp16x2,
    int32_t verbosity, const udevptrs<float> &samples,
    const udevptrs<float> &centroids, const udevptrs<uint32_t> &assignments,
    float *average_distance) {
  INFO("calculating the average distance...\n");
  auto plans = distribute(h_samples_size, h_features_size * sizeof(float), devs);
  udevptrs<atomic_float> dev_dists;
  CUMALLOC(dev_dists, sizeof(atomic_float));
  CUMEMSET_ASYNC(dev_dists, 0, sizeof(atomic_float));
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plans[devi];
    if (length == 0) {
      continue;
    }
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid(upper(length, block.x), 1, 1);
    KERNEL_SWITCH(kmeans_calc_average_distance,
                  <<<grid, block, block.x * sizeof(float)>>>(
        offset, length, reinterpret_cast<const F*>(samples[devi].get()),
        reinterpret_cast<const F*>(centroids[devi].get()),
        assignments[devi].get(), dev_dists[devi].get()));
  );
  atomic_float sum = 0;
  FOR_EACH_DEVI(
    atomic_float hdist;
    CUCH(cudaMemcpy(&hdist, dev_dists[devi].get(), sizeof(atomic_float),
                    cudaMemcpyDeviceToHost),
         kmcudaMemoryCopyError);
    sum += hdist;
  );
  *average_distance = sum / h_samples_size;
  return kmcudaSuccess;
}


}  // extern "C"
