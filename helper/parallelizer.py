from threading import Semaphore, Thread
from typing import Any, Callable, Dict, Iterable, List


def parallelize_task_and_return_results(task: Callable, args: List[Any], thread_sem: Semaphore, result_storage_sem: Semaphore,
                                        thread_number: int, result_storage: Dict[int, Any]) -> None:
    """
        Execute a task with a thread and save result in an on reference passed storage/variable.
    """

    result = task(*args)  # Store result
    result_storage_sem.acquire()
    result_storage[thread_number] = result
    result_storage_sem.release()
    thread_sem.release()  # Next thread can start


def parallelize_task_with_return_values(task: Callable, args: Iterable, max_number_of_runnings_threads: int = 8, print_iteration: int=1000) -> Dict[int, Any]:
    """
        Executes a passed function "task" with passed arguments "args"
        and returns results of each thread as list.
        If no iteration print is desired, then set "print_iteration = 0" or
        "print_iteration = None".
    """

    # results = []  # Results of each thread
    # pool = ThreadPool(max_number_of_runnings_threads)

    # # Add all tasks and create threads
    # for args_per_function_call in args:
    #     args_per_function_call = args_per_function_call if isinstance(args_per_function_call, Iterable) else (args_per_function_call)
    #     results.append(pool.apply_async(task, args=args_per_function_call))

    # # Wait for results of the pool
    # pool.close()
    # pool.join()
    # results = [r.get() for r in results]
    # return results

    threads, results = [], {}
    thread_sem = Semaphore(max_number_of_runnings_threads)
    result_storage_sem = Semaphore(1)


    # Compute tasks parallilized
    for i, args_per_function_call in enumerate(args):
        if i % print_iteration == 0:
            print(f"Iteration {i}")

        thread_sem.acquire()
        args = [task, args_per_function_call, thread_sem, result_storage_sem, i, results]  # Args of function task
        thread = Thread(target=parallelize_task_and_return_results, args=args)
        threads.append(thread)
        thread.start()

    # Wait for results
    for thread in threads:
        thread.join()

    return results


def parallelize_task(task: Callable, args: List[Any], thread_sem: Semaphore, thread_number: int) -> None:
    """
        Execute parallely a task with a thread.
    """

    task(*args)  # Do task
    thread_sem.release()  # Next thread can start


def parallelize_task_without_return_values(task: Callable, args: Iterable, max_number_of_runnings_threads: int = 8, print_iteration: int=1000):
    """
        Executes parallely a passed function "task" with passed arguments "args".
        If no iteration print is desired, then set "print_iteration = 0" or
        "print_iteration = None".
    """

    # results = []  # Results of each thread
    # pool = ThreadPool(max_number_of_runnings_threads)

    # # Add all tasks and create threads
    # for args_per_function_call in args:
    #     args_per_function_call = args_per_function_call if isinstance(args_per_function_call, Iterable) else (args_per_function_call)
    #     results.append(pool.apply_async(task, args=args_per_function_call))

    # # Wait for results of the pool
    # pool.close()
    # pool.join()
    # results = [r.get() for r in results]
    # return results

    threads = []
    thread_sem = Semaphore(max_number_of_runnings_threads)

    # Compute tasks parallilized
    for i, args_per_function_call in enumerate(args):
        if print_iteration and i % print_iteration == 0:
            print(f"Iteration {i}")

        thread_sem.acquire()
        args = [task, args_per_function_call, thread_sem, i]  # Args of function task
        thread = Thread(target=parallelize_task, args=args)
        threads.append(thread)
        thread.start()

    # Wait for results
    for thread in threads:
        thread.join()
