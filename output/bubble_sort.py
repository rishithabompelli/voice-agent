def bubble_sort(arr):
    """
    Sorts an array in ascending order using the Bubble Sort algorithm.

    Args:
        arr (list): The input list to be sorted.

    Returns:
        list: The sorted list.
    """

    # Get the length of the input array
    n = len(arr)

    # Repeat the process until no more swaps are needed
    for i in range(n):
        
        # Initialize a flag to track if any swaps were made
        swapped = False
        
        # Iterate through each element in the unsorted portion of the array
        for j in range(n - i - 1):
            
            # If the current element is greater than the next one, swap them
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        # If no swaps were made in this pass, the array is already sorted
        if not swapped:
            break
    
    return arr

# Example usage:
if __name__ == "__main__":
    arr = [64, 34, 25, 12, 22, 11, 90]
    print("Original array:", arr)
    sorted_arr = bubble_sort(arr)
    print("Sorted array:", sorted_arr)

This code implements the Bubble Sort algorithm to sort an input list in ascending order. It uses two nested loops to compare each pair of adjacent elements and swap them if they are out of order. The process is repeated until no more swaps are needed, indicating that the array is already sorted.