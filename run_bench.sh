BATCH_PER_GPU=64
NUM_GPU=8
CONCURRENCY=$((BATCH_PER_GPU * NUM_GPU))

# uncomment following lines to capture nsys profile

# TLLM_PROFILE_START_STOP=200-220 nsys profile \
#  -o profile_6k_64batch_gen -f true -t 'cuda,nvtx,python-gil' -c cudaProfilerApi --cuda-graph-trace node \
#  -e TLLM_PROFILE_RECORD_GC=1,TLLM_LLMAPI_ENABLE_NVTX=1,TLLM_TORCH_PROFILE_TRACE=trace.json --trace-fork-before-exec=true \
trtllm-bench \
 -m /workspaces/tensorrt_llm/hf-ckpt \
 --model_path /workspaces/tensorrt_llm/hf-ckpt \
 throughput \
 --tp $NUM_GPU --ep $NUM_GPU --warmup 0 \
 --dataset dataset4k.txt \
 --backend pytorch \
 --max_batch_size $CONCURRENCY --max_num_tokens 5159 --num_requests $CONCURRENCY --concurrency $CONCURRENCY \
 --kv_cache_free_gpu_mem_fraction 0.8 \
 --extra_llm_api_options ./extra-llm-api-config.yml \
 2>&1 | tee log_4k512b_3k_debug_2.txt

# python parse_iter_log.py --file log_6k512b_3k.txt --concurrency 512 --enable_dp --gpu_num 8
