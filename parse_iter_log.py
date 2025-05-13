import re


def parse_log(file: str, concurrency: int, enable_dp: bool, gpu_num: int = 8):
    batch_per_gpu = concurrency // gpu_num if enable_dp else concurrency
    pattern = re.compile(
        f"elapsed_time = (.+?)s.+num_scheduled_requests: {batch_per_gpu}.+'num_generation_tokens': {batch_per_gpu}"
    )
    time_list = []
    with open(file, 'r') as f:
        skip = True  # only collect continuous pure-generation iterations
        for line in f:
            match = pattern.search(line)
            if match:
                if skip:
                    skip = False
                else:
                    elapsed_time = float(match.group(1))
                    time_list.append(elapsed_time)
            else:
                skip = True

    total_time = sum(time_list)
    iters = len(time_list)
    total_tokens = concurrency * iters
    tps = total_tokens / total_time
    tps_per_gpu = tps / gpu_num
    tps_per_user = tps / concurrency

    print(
        f"There are {iters} iterations with full {batch_per_gpu} generation phase."
    )
    # print(time_list)
    print(f"Average iteration time: {total_time / iters:.6f} s")
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
