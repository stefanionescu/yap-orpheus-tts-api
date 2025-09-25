# TTS server defaults - optimized for lower TTFB
export FIRST_CHUNK_WORDS=${FIRST_CHUNK_WORDS:-18}    # Reduced from 40 for faster TTFB
export NEXT_CHUNK_WORDS=${NEXT_CHUNK_WORDS:-120}     # Slightly reduced from 140
export MIN_TAIL_WORDS=${MIN_TAIL_WORDS:-12}
export SNAC_TORCH_COMPILE=${SNAC_TORCH_COMPILE:-0}
export SNAC_MAX_BATCH=${SNAC_MAX_BATCH:-64}
export SNAC_BATCH_TIMEOUT_MS=${SNAC_BATCH_TIMEOUT_MS:-2}
export ORPHEUS_MAX_TOKENS=${ORPHEUS_MAX_TOKENS:-2048}

# Logging configuration - enable DEBUG level by default for comprehensive monitoring
export LOG_LEVEL=${LOG_LEVEL:-DEBUG}

# Audio token thresholds - optimized for lower TTFB  
export MIN_TOKENS_FIRST=${MIN_TOKENS_FIRST:-28}      # Reduced from 56 for faster first audio
export MIN_TOKENS_SUBSEQ=${MIN_TOKENS_SUBSEQ:-28}    # Keep at 28 for quality
export TOKENS_EVERY=${TOKENS_EVERY:-7}               # Tokens per SNAC frame (7 is canonical for Orpheus)
# Backward-compat names (if users still set frames-based vars)
export MIN_FRAMES_FIRST=${MIN_FRAMES_FIRST:-14}
export MIN_FRAMES_SUBSEQ=${MIN_FRAMES_SUBSEQ:-28}
export PROCESS_EVERY=${PROCESS_EVERY:-7}

# Performance optimizations
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-2}
export ORPHEUS_MAX_SECONDS=${ORPHEUS_MAX_SECONDS:-10}
