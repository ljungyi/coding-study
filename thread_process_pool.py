from concurrent.futures import (
    ThreadPoolExecutor,ProcessPoolExecutor)


from time import perf_counter,sleep

def cpu(n):
    total = 0
    for i in range(n):
        total += i*i
    return total

def io(n):
    sleep(1)
    return 1

def benchmark(poolexe,func,data):
    start=perf_counter()
    with poolexe(max_workers=4) as pool:
        result=list(pool.map(func,data))
    elapsed=perf_counter()-start
    return elapsed,result


if __name__ == "__main__":

    print("=== I/O task ===")

    t_thread, _ = benchmark(
        ThreadPoolExecutor,
        io,
        range(4)
    )

    t_process, _ = benchmark(
        ProcessPoolExecutor,
        io,
        range(4)
    )

    print("thread :", t_thread)
    print("process:", t_process)

    print("\n=== CPU task ===")

    values = [10_000_000] * 4

    t_thread, _ = benchmark(
        ThreadPoolExecutor,
        cpu,
        values
    )

    t_process, _ = benchmark(
        ProcessPoolExecutor,
        cpu,
        values
    )

    print("thread :", t_thread)
    print("process:", t_process)


