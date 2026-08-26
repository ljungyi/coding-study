from functools import wraps

def retry(max_retries=3,exceptions=(Exception,)):
    def decorator(fn):
        def wrapper(*args,**kwargs):
            attempts = 0
            while attempts < max_retries:
                try:
                    return fn(*args,**kwargs)
                except exceptions as e:
                    print(f"attempt {attempts+1} failed")
                    attempts += 1
                    if attempts == max_retries:
                        raise e  
        return wrapper
    return decorator


counter = 0


@retry(
    3,
    ValueError
)
def unstable():

    global counter
    counter += 1

    if counter < 3:
        raise ValueError("temporary failure")

    return "success"


print(unstable())
               
        