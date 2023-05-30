import time
import threading

# Define a shared list
my_list = [1]

# Define a thread function to append items to the list
def append_items():
    for i in range(5):
        suma = sum(my_list) + 1
        my_list.append(suma)
        threading.current_thread().setName(i)
        print(f"{threading.current_thread().getName()} Appended {suma} to the list")
# Define a thread function to remove items from the list
def remove_items():
    for i in range(3):
        if my_list:
            item = my_list.pop()
            print(f"Removed {item} from the list")
        else:
            print("List is empty")

# Create and start the threads
append_thread = threading.Thread(target=append_items)
# remove_thread = threading.Thread(target=remove_items)
start_time = time.time()

# append_thread.start()
# append_thread.join()
append_items()
# remove_thread.start()

# Wait for both threads to finish
end_time = time.time()

print(f'elapsed time: {end_time - start_time}s')
# remove_thread.join()

# Print the final state of the list
print("Final list:", my_list)
