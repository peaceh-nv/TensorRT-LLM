BATCH_PER_GPU=64
NUM_GPU=8
CONCURRENCY=$((BATCH_PER_GPU * NUM_GPU))

# uncomment following lines to capture nsys profile

# TLLM_PROFILE_START_STOP=900-920 nsys profile \
#  -o profile_6k_64batch -f true -t 'cuda,nvtx,python-gil' -c cudaProfilerApi --cuda-graph-trace node \
#  -e TLLM_PROFILE_RECORD_GC=1,TLLM_LLMAPI_ENABLE_NVTX=1,TLLM_TORCH_PROFILE_TRACE=trace.json --trace-fork-before-exec=true \
trtllm-bench \
 -m /home/scratch.xiweny_gpu_1/envs/sm120/dsr1-fp4 \
 throughput \
 --tp $NUM_GPU --ep $NUM_GPU --warmup 0 \
 --dataset dataset6k.txt \
 --backend pytorch \
 --max_batch_size $CONCURRENCY --max_num_tokens 7220 --num_requests $CONCURRENCY --concurrency $CONCURRENCY \
 --kv_cache_free_gpu_mem_fraction 0.93 \
 --extra_llm_api_options ./extra-llm-api-config.yml \
 | tee log_6k64b.txt

python parse_iter_log.py --file log_6k64b.txt --concurrency $CONCURRENCY --enable_dp --gpu_num $NUM_GPU
