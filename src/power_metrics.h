/*
 * Declaration header added in the local Futurism development history.
 * The referenced power-measurement helpers already exist in Kubism.
 * This header is absent from both pinned upstream source trees.
 * Pinned sources and file comparisons: docs/SOURCE_AUDIT.md.
 * Source attribution and license scope: docs/PROVENANCE.md.
 */

// ...existing code...
#ifndef POWER_METRICS_H
#define POWER_METRICS_H

#ifdef __cplusplus
extern "C" {
#endif

void capture_power_metrics_in_interval_start(int iteration);
void capture_power_metrics_in_interval_end(int iteration);

#ifdef __cplusplus
}
#endif

#endif // POWER_METRICS_H
// ...existing code...