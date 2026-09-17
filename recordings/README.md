# Recordings

Missions are recorded here automatically as JSON: a flat, time-stamped list of
detection and pose frames plus the mission summary.

A replay pushes those frames back through the *same* pipeline live data uses, so
the occupancy grid, surface fits and statistics are recomputed from an empty
canvas rather than replayed as pictures. It needs no phone, no microphone and
no network — which is why this directory is the demo's insurance policy.

**Before presenting: do a good run and keep it.** If the live demo collapses,
`REPLAY LAST` on the command center is a complete substitute that tells the
whole story.

Any file named `reference-*.json` is a deliberately kept known-good scan.
