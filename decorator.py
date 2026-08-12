from contextlib import contextmanager
from time import perf_counter

@contextmanager
def time_count(name:str="default"):
    start=perf_counter()
    try:
        yield
    finally:
        end=perf_counter()
        time=end-start
        print(f"{name} consumes {time:.6f}")

with time_count("count"):
    range(10000000000000000000000000000);
