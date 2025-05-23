import re

time_list_ctx = []
time_list_gen = []


def parse_log(file: str, concurrency: int, enable_dp: bool, gpu_num: int = 8):
    batch_per_gpu = concurrency // gpu_num if enable_dp else concurrency
    pattern = re.compile(
        f"elapsed_time = (.+?)s.+num_scheduled_requests: 47.+'num_generation_tokens': 64"
    )
    pattern_ctx = re.compile(f"elapsed_time = (.+?)s.+'num_ctx_requests': 1")
    with open(file, 'r') as f:
        skip = True  # only collect continuous pure-generation iterations
        for line in f:
            match_ctx = pattern_ctx.search(line)
            if match_ctx:
                elapsed_time = float(match_ctx.group(1))
                time_list_ctx.append(elapsed_time)
            match = pattern.search(line)
            if match:
                if skip:
                    skip = False
                else:
                    elapsed_time = float(match.group(1))
                    time_list_gen.append(elapsed_time)
            else:
                skip = True

    total_time_gen = sum(time_list_gen)
    iters_gen = len(time_list_gen)
    mean_gen_time = total_time_gen / iters_gen
    std_var_gen = sum(
        (time - mean_gen_time)**2 for time in time_list_gen) / iters_gen
    mean_ctx_time = sum(time_list_ctx) / len(time_list_ctx)
    std_var_ctx = sum((time - mean_ctx_time)**2
                      for time in time_list_ctx) / len(time_list_ctx)
    total_tokens = concurrency * iters_gen
    tps = total_tokens / total_time_gen
    tps_per_gpu = tps / gpu_num
    tps_per_user = tps / concurrency

    print(f"Mean gen time: {mean_gen_time:.6f} s")
    print(f"Standard gen deviation: {std_var_gen:.6f} s")
    print(f"Mean ctx time: {mean_ctx_time:.6f} s")
    print(f"Standard ctx deviation: {std_var_ctx:.6f} s")
    print(
        f"There are {iters_gen} iterations with full {batch_per_gpu} generation phase."
    )
    print(f"Average iteration time: {total_time_gen / iters_gen:.6f} s")
    print(f"TPS: {tps:.2f} tokens/s")
    print(f"TPS per GPU: {tps_per_gpu:.2f} tokens/s")
    print(f"TPS per user: {tps_per_user:.2f} tokens/s")


# parse_log(file='log_6k64b.txt', concurrency=64, enable_dp=True, gpu_num=8)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Parse log file.')
    parser.add_argument('--file',
                        type=str,
                        required=True,
                        help='Path to the log file')
    parser.add_argument('--concurrency',
                        type=int,
                        required=True,
                        help='Concurrency level')
    parser.add_argument('--enable_dp',
                        action='store_true',
                        help='Enable data parallelism')
    parser.add_argument('--gpu_num', type=int, default=8, help='Number of GPUs')

    args = parser.parse_args()

    parse_log(args.file, args.concurrency, args.enable_dp, args.gpu_num)
