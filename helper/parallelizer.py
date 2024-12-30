import traceback

from threading import Semaphore, Thread
from typing import Any, Callable, Dict, Iterable, List


class ThreadPool:
    def __init__(self, max_number_of_runnings_threads: int = 8):
        self.__max_number_of_runnings_threads = max_number_of_runnings_threads

    def _do_task_and_return_results(
        self,
        task: Callable,
        all_args: List[Any],
        result_storage_sem: Semaphore,
        result_storage: Dict[int, Any],
        thread_number: int,
        current_iteration_sem: Semaphore,
        current_iteration: Dict[str, float],
        print_iteration: int,
    ) -> None:
        """
        Executes passed task with passed arguemnts sequentially. Results will
        be stored in a list and added to a result storage "result_storage"
        accessable for all threads, but access will be sysnchronized with
        "result_storage_sem".

        Parameters
        ----------
        task : Callable
            Task/Function to execute with passed arguments
        all_args : List[Any]
            Contains all arguments a thread must pass the function task
        result_storage_sem : Semaphore
            Semaphore for synchronizing access to variable "result_storage"
        result_storage : Dict[int, Any]
            Storage for saving results of for threads
        thread_number : int
            Number of a thread
        current_iteration_sem : Semaphore
            Semaphore for synchronizing access to variable "current_iteration"
        current_iteration : Dict[str, float]
            Object containing information about the current information and when to update
        print_iteration : int
            Check for updating and eventually printing current iteration
        """

        results = []

        # Execute for each args a function an save results in temporary list.
        for i, args in enumerate(all_args):
            try:
                if i % print_iteration == 0:  # Update number iterations and eventually output it
                    current_iteration_sem.acquire()
                    # Update number of iterations
                    if 0 < i:
                        current_iteration["current_iteration"] += print_iteration

                    # Output current iteration of all threads
                    if (
                        current_iteration["next_iteration_to_output"] <= current_iteration["current_iteration"]
                    ):  # Next it <= current it
                        iteration_to_output = (
                            current_iteration["current_iteration"] // current_iteration["output_after_n_iterations"]
                        )
                        iteration_to_output *= current_iteration["output_after_n_iterations"]
                        current_iteration["next_iteration_to_output"] += current_iteration[
                            "output_after_n_iterations"
                        ]  # Increase next output iteration
                        print(f"Iteration {iteration_to_output}")
                    current_iteration_sem.release()

                results.append(task(*args))  # Store result
            except KeyboardInterrupt:
                exit()
            except Exception:
                print(f"Thread {thread_number} has an error:")
                print(traceback.format_exc(), "\n")

        # Save results in overall dict
        result_storage_sem.acquire()
        result_storage[thread_number] = results
        result_storage_sem.release()

    def join(
        self,
        task: Callable,
        args: Iterable,
        print_iteration: int = 1000,
        no_print: bool = False
    ) -> List[Any]:
        """
        Executes a passed function "task" with passed arguments "args"
        and returns results of each thread summarized in a list.\n
        The results will be sorted after each thread and each thread executes
        it's function sequentially. Overall the results will be sorted same as
        calling the function "task" sequentially with the passed arguments.\n
        
        If no iteration print is desired, then set "print_iteration = 0" or
        "print_iteration = None".

                task : Callable
            Task/Function to execute with passed arguments
        all_args : List[Any]
            Contains all arguments a thread must pass the function task
        result_storage_sem : Semaphore

        Parameters
        ----------
        task : Callable
            Task/Function to execute with passed arguments
        args: Iterable,
            Contains all arguments for all threads, so that each thread can
            call the function "task" with it's passed parts of arguments
            "args"
        print_iteration : int, default 1000
            Print after "print_iteration" iterations the current number of
            iterations
        no_print : bool, default False
            If True then no iterations will be outputted, else iterations will
            be outputted depending on variable "print_iteration"
        """

        # Define variables
        threads, results = [], {}
        current_iteration = {
            "current_iteration": 0,
            "next_iteration_to_output": 0,
            "output_after_n_iterations": print_iteration,
        }
        result_storage_sem = Semaphore(1)
        current_iteration_sem = Semaphore(1)

        # Split arguments so that each thread has more or less an equal amount of tasks to do
        n_args_per_thread = len(args) // self.__max_number_of_runnings_threads + 1  # "+1" because of really doing all tasks
        args_for_all_threads = [args[i:i + n_args_per_thread] for i in range(0, len(args), n_args_per_thread)]

        # Define, when a thread should print it's iteration
        if no_print or not print_iteration:
            prints_per_threads = len(args) + 1  # Set first print higher than amount of all arguments
            current_iteration["output_after_n_iterations"] = prints_per_threads
        else:
            prints_per_threads = int(print_iteration / self.__max_number_of_runnings_threads)
            prints_per_threads = 1 if prints_per_threads == 0 else prints_per_threads  # Print at least after one iteration

        # Compute tasks parallelized
        for i, args_per_thread in enumerate(args_for_all_threads):
            args = [
                task,
                args_per_thread,
                result_storage_sem,
                results,
                i,
                current_iteration_sem,
                current_iteration,
                prints_per_threads,
            ]  # Args of function "task"
            thread = Thread(target=self._do_task_and_return_results, args=args)
            threads.append(thread)
            thread.start()

        # Wait for results
        for i, thread in enumerate(threads):
            thread.join()

        # Sort results
        if not no_print:
            print("Sort results")
        results = dict(list(sorted(results.items())))

        # Reformat results
        if not no_print:
            print("Reformat results")
        results = results.values()  # Thread number is irrelevant for caller
        formatted_results = []

        for results_of_a_thread in results:
            formatted_results.extend(results_of_a_thread)

        return formatted_results  # Return results
