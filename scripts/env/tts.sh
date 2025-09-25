# TTS server defaults - aggressively optimized for TTFB
export FIRST_CHUNK_WORDS=${FIRST_CHUNK_WORDS:-16}    # Very small first chunk for minimal prefill
export NEXT_CHUNK_WORDS=${NEXT_CHUNK_WORDS:-120}     # Keep reasonable for subsequent chunks
export MIN_TAIL_WORDS=${MIN_TAIL_WORDS:-12}
export SNAC_TORCH_COMPILE=${SNAC_TORCH_COMPILE:-1}      # Enable compilation for better perf
export SNAC_MAX_BATCH=${SNAC_MAX_BATCH:-64}
export SNAC_BATCH_TIMEOUT_MS=${SNAC_BATCH_TIMEOUT_MS:-2}  # Aggressive batching for latency
export ORPHEUS_MAX_TOKENS=${ORPHEUS_MAX_TOKENS:-2048}

# Logging configuration - enable DEBUG level by default for comprehensive monitoring
export LOG_LEVEL=${LOG_LEVEL:-DEBUG}

# Audio token thresholds - AGGRESSIVELY optimized for TTFB  
export MIN_TOKENS_FIRST=${MIN_TOKENS_FIRST:-7}       # 1 frame only! Minimal latency
export MIN_TOKENS_SUBSEQ=${MIN_TOKENS_SUBSEQ:-28}    # Keep at 28 for quality (4 frames)
export TOKENS_EVERY=${TOKENS_EVERY:-7}               # Tokens per SNAC frame (7 is canonical for Orpheus)
# SNAC lane order - try different orders if you hear gunshot/static artifacts
export ORPHEUS_LANE_ORDER=${ORPHEUS_LANE_ORDER:-"0|1,2|3,4,5,6"}  # contiguous (default)
# Alternative orders to try: "0|1,4|2,3,5,6" or "0|1,5|2,3,4,6" or "0|2,3|1,4,5,6"

# Backward-compat names (if users still set frames-based vars)
export MIN_FRAMES_FIRST=${MIN_FRAMES_FIRST:-14}
export MIN_FRAMES_SUBSEQ=${MIN_FRAMES_SUBSEQ:-28}
export PROCESS_EVERY=${PROCESS_EVERY:-7}

# Performance optimizations
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-2}
export ORPHEUS_MAX_SECONDS=${ORPHEUS_MAX_SECONDS:-10}
