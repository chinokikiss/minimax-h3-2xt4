# Sol All-INT8 SM75 experiment

This directory contains an isolated T4 experiment derived from the Star7 Sol
All-INT8 implementation. It does not modify production attention dispatch.

The benchmark compares complete raw-input calls at identical FP16 BHND geometry:
Star7-derived Sol All-INT8 with tau=1.0 versus comfy-kitchen 0.2.37 dense INT8.
Routing, quantization, centroid construction, V quantization, output allocation,
and attention execution are included in Sol timing. Kitchen timing includes its
normal complete int8_attention call. Synthetic inputs are used; this is operator
throughput evidence, not generation-quality evidence.

Vendored Star7 source remains under the upstream MIT license. The copied Kitchen
and SageAttention source retains its Apache-2.0 license files.
