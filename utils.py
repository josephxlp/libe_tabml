import os
import time 

def print_context(text):
    print(' ')
    """
    Prints the given text in a styled format, mimicking SAGA GIS console output.

    Parameters:
        text (str): The text to print.
    """
    # Border symbols
    border_char = "=" * 60
    padding_char = " " * 4

    # Print formatted text
    print(border_char)
    print(f"{padding_char}ML - Process Log")
    print(border_char)

    for line in text.split("\n"):
        print(f"{padding_char}{line}")

    print(border_char)


def notify_send(title: str, message: str, duration: int = 5):
    """
    Displays a notification on Linux using notify-send.
    
    Parameters:
    title (str): The notification title.
    message (str): The notification message.
    duration (int): Time in seconds to display the notification.
    """
    os.system(f'notify-send -t {duration * 1000} "{title}" "{message}"')

def measure_time_beautifully(task_description, task_function, *args, **kwargs):
    """
    Measures the execution time of a given function and prints the elapsed time in various units.

    Parameters:
        task_description (str): A description of the task being measured.
        task_function (callable): The function to execute and measure.
        *args: Positional arguments to pass to the task_function.
        **kwargs: Keyword arguments to pass to the task_function.
    """
    # Start the timer
    start_time = time.perf_counter()

    # Execute the task function
    result = task_function(*args, **kwargs)

    # Stop the timer
    end_time = time.perf_counter()

    # Calculate elapsed time
    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60
    elapsed_hours = elapsed_minutes / 60
    elapsed_days = elapsed_hours / 24

    # Border symbols
    border_char = "=" * 60
    padding_char = " " * 4

    # Print the execution time beautifully
    print(border_char)
    print(f"{padding_char}Task Performance Report")
    print(border_char)
    print(f"{padding_char}Task: {task_description}")
    print(f"{padding_char}Elapsed Time:")
    print(f"{padding_char * 2}{elapsed_seconds:.2f} seconds")
    print(f"{padding_char * 2}{elapsed_minutes:.2f} minutes")
    print(f"{padding_char * 2}{elapsed_hours:.2f} hours")
    print(f"{padding_char * 2}{elapsed_days:.2f} days")
    print(border_char)

    return result




