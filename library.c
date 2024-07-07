#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#define CHUNK_SIZE 1024 * 1024 // 1 MB
#define MAX_KEYWORD_LENGTH 256
#define MAX_FILENAME_LENGTH 1024


/*  CUDA KERNAL:
*  Each thread checks a portion of the buffer for the keyword. Marks the positions where the keyword is foun.
 *  This function is the CUDA kernel that searches for a keyword in a buffer.
 *  The kernel is launched with a grid of blocks, where each block has a number of threads.
 *  Each thread searches for the keyword in a portion of the buffer.
 *  If the keyword is found, the corresponding result is set to 1.
 *  The results are stored in a separate array.
 */


__global__ void search_kernel(char *buffer, size_t buffer_size, char *keyword, size_t keyword_length, int *results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx + keyword_length <= buffer_size) {
        if (strncmp(&buffer[idx], keyword, keyword_length) == 0) {
            results[idx] = 1;
        } else {
            results[idx] = 0;
        }
    }
}

void search_in_chunk(char *buffer, size_t buffer_size, char *keyword, size_t keyword_length, size_t chunk_start) {
    char *d_buffer;
    char *d_keyword;
    int *d_results;
    int *h_results = (int *)malloc(buffer_size * sizeof(int));
    size_t keyword_bytes = keyword_length * sizeof(char);

    printf("Allocating device memory...\n");
    cudaMalloc((void **)&d_buffer, buffer_size);
    cudaMalloc((void **)&d_keyword, keyword_bytes);
    cudaMalloc((void **)&d_results, buffer_size * sizeof(int));

    printf("Copying data to device...\n");
    cudaMemcpy(d_buffer, buffer, buffer_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_keyword, keyword, keyword_bytes, cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (buffer_size + block_size - 1) / block_size;

    printf("Launching kernel with %d blocks of %d threads each...\n", num_blocks, block_size);
    search_kernel<<<num_blocks, block_size>>>(d_buffer, buffer_size, d_keyword, keyword_length, d_results);

    printf("Copying results back to host...\n");
    cudaMemcpy(h_results, d_results, buffer_size * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Processing results...\n");
    for (size_t i = 0; i < buffer_size; i++) {
        if (h_results[i] == 1) {
            printf("Keyword found at position: %zu\n", chunk_start + i);
        }
    }

    printf("Freeing device memory...\n");
    cudaFree(d_buffer);
    cudaFree(d_keyword);
    cudaFree(d_results);
    free(h_results);
}

/* MAIN FUNCTION:
 * The main function reads the keyword and filename from the user. Then it opens the file and reads it in chunks and offloads the
 *  heavy lifting to the GPU using the 'search kernal'
 *  the 'search kernal' is launched with a grid of blocks, where each memory block has a number of threads for the buffer, keywords
 *  and results.
 *  Matching data is copied between
 */
int main() {
    char keyword[MAX_KEYWORD_LENGTH];
    char filename[MAX_FILENAME_LENGTH];

    // Get the keyword from the user
    printf("Enter the keyword to search: ");
    if (fgets(keyword, MAX_KEYWORD_LENGTH, stdin) == NULL) {
        perror("Error reading keyword");
        return EXIT_FAILURE;
    }
    // Remove newline character if present
    size_t keyword_length = strlen(keyword);
    if (keyword[keyword_length - 1] == '\n') {
        keyword[keyword_length - 1] = '\0';
        keyword_length--;
    }

    // Get the filename from the user
    printf("Enter the filename to search in: ");
    if (fgets(filename, MAX_FILENAME_LENGTH, stdin) == NULL) {
        perror("Error reading filename");
        return EXIT_FAILURE;
    }
    // Remove newline character if present
    size_t filename_length = strlen(filename);
    if (filename[filename_length - 1] == '\n') {
        filename[filename_length - 1] = '\0';
    }

    printf("Opening file: %s\n", filename);
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return EXIT_FAILURE;
    }

    char *buffer = (char *)malloc(CHUNK_SIZE + keyword_length - 1);
    if (buffer == NULL) {
        perror("Error allocating buffer");
        fclose(file);
        return EXIT_FAILURE;
    }

    size_t chunk_start = 0;
    size_t bytes_read;
    while ((bytes_read = fread(buffer, 1, CHUNK_SIZE, file)) > 0) {
        printf("Read %zu bytes from file...\n", bytes_read);
        buffer[bytes_read] = '\0'; // Null-terminate the buffer
        search_in_chunk(buffer, bytes_read, keyword, keyword_length, chunk_start);
        chunk_start += bytes_read;

        // Move the buffer's start to the position after the last overlap
        fseek(file, -(long)(keyword_length - 1), SEEK_CUR);
        chunk_start -= keyword_length - 1;
    }

    printf("Freeing host buffer memory...\n");
    free(buffer);
    fclose(file);

    printf("Search complete.\n");
    return EXIT_SUCCESS;
}
