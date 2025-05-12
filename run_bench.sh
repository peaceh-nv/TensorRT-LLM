# uncomment following lines to capture nsys profile

# TLLM_PROFILE_START_STOP=1000-1020 nsys profile \
#  -o profile_6kb8 -f true -t 'cuda,nvtx,python-gil' -c cudaProfilerApi --cuda-graph-trace node \
#  -e TLLM_PROFILE_RECORD_GC=1,TLLM_LLMAPI_ENABLE_NVTX=1,TLLM_TORCH_PROFILE_TRACE=trace.json --trace-fork-before-exec=true \
trtllm-bench \
 -m /home/scratch.trt_llm_data/llm-models/DeepSeek-R1/DeepSeek-R1-FP4 \
 throughput \
 --tp 8 --ep 8 --warmup 0 \
 --dataset dataset6k.txt \
 --backend pytorch \
 --max_batch_size 64 --max_num_tokens 7300 --num_requests 192 --concurrency 64 \
 --kv_cache_free_gpu_mem_fraction 0.8 \
 --extra_llm_api_options ./extra-llm-api-config.yml \
 | tee log_6k64b.txt

python parse_iter_log.py --file log_6k64b.txt --concurrency 64 --enable_dp --gpu_num 8
