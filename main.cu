#include <chrono>
#include <cstdio>
#include <iostream>

#define BLOCK_SIZE 512
#define NUM_BLOCKS 45
// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 2048

// Graph structure
typedef struct {
  int numNodes;  // number of nodes
  int *nodePtrs; // each int represents a position in the nodeNeighbors array:
                 // it's the first neighbor of node i (the last one is
                 // nodePtrs[i+1]-1)
  int *nodeNeighbors; // each int represents a neighbor of the node i as above
                      // explained
} Graph;

Graph *loadGraphUndirected(const char *filename) {
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    perror("Error opening file");
    return NULL;
  }

  int numNodes = 0;
  fscanf(file, "%d", &numNodes);
  bool **matrix_graph = (bool **)calloc(numNodes, sizeof(bool *));

  // start by filling the matrix with zeros
  for (int i = 0; i < numNodes; i++) {
    matrix_graph[i] = (bool *)calloc(numNodes, sizeof(bool));
  }

  // scan total number of edges reading it from the file and dump it
  int numEdges;
  fscanf(file, "%d", &numEdges);

  // read the file and fill the matrix
  int sourceNode = 0;
  int destNode = 0;
  while (fscanf(file, "%d", &sourceNode) != EOF) {
    fscanf(file, "%d", &destNode);
    matrix_graph[sourceNode][destNode] = true;
    matrix_graph[destNode][sourceNode] = true;
  }

  // now we can count the number of edges
  numEdges = 0;
  for (int i = 0; i < numNodes; i++) {
    for (int j = 0; j < numNodes; j++) {
      if (matrix_graph[i][j]) {
        numEdges++;
      }
    }
  }

  // allocate memory for the graph with cudaMallocManaged
  Graph *graph;
  cudaMallocManaged(&graph, sizeof(Graph));

  // set number of nodes
  graph->numNodes = numNodes;

  // allocate memory for nodePtrs and nodeNeighbors with cudaMallocManaged
  cudaMalloc(&graph->nodePtrs, sizeof(int) * (graph->numNodes + 1));
  cudaMalloc(&graph->nodeNeighbors, sizeof(int) * numEdges);

  // allocate memory for nodePtrsHost and nodeNeighborsHost with malloc
  int *nodePtrsHost = (int *)malloc(sizeof(int) * (graph->numNodes + 1));
  int *nodeNeighborsHost = (int *)malloc(sizeof(int) * numEdges);

  // fill nodePtrs and nodeNeighbors arrays
  int edgeIdx = 0;
  nodePtrsHost[0] = 0;
  for (int i = 0; i < numNodes; i++) {
    for (int j = 0; j < numNodes; j++) {
      if (matrix_graph[i][j]) {
        nodeNeighborsHost[edgeIdx] = j;
        edgeIdx++;
      }
    }
    nodePtrsHost[i + 1] = edgeIdx;
  }

  // copy nodePtrs and nodeNeighbors arrays to device
  cudaMemcpy(graph->nodePtrs, nodePtrsHost, sizeof(int) * (graph->numNodes + 1),
             cudaMemcpyHostToDevice);
  cudaMemcpy(graph->nodeNeighbors, nodeNeighborsHost, sizeof(int) * numEdges,
             cudaMemcpyHostToDevice);

  // free memory
  for (int i = 0; i < numNodes; i++) {
    free(matrix_graph[i]);
  }

  free(matrix_graph);

  fclose(file);

  return graph;
}

// Global queuing stub
__global__ void gpu_global_queuing_kernel(int *nodePtrs, int *nodeNeighbors,
                                          int *nodeVisited, int *currLevelNodes,
                                          int *nextLevelNodes,
                                          const int numCurrLevelNodes,
                                          int *numNextLevelNodes) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  // Iterate over the nodes in the current level. The loop stride is the total
  // number of threads.
  for (int i = idx; i < numCurrLevelNodes; i += stride) {
    // Get the node at the current index.
    int node = currLevelNodes[i];
    for (int j = nodePtrs[node]; j < nodePtrs[node + 1]; j++) {
      // Get the neighbor at the current index.
      int neighbor = nodeNeighbors[j];
      // If the neighbor has not been visited yet.
      if (nodeVisited[neighbor] == 0) {
        // Mark the neighbor as visited.
        nodeVisited[neighbor] = 1;
        // Add the neighbor to the list of nodes to visit in the next level.
        // The atomicAdd function ensures that this operation is thread-safe.
        nextLevelNodes[atomicAdd(numNextLevelNodes, 1)] = neighbor;
      }
    }
  }
}

int main() {
  const char *filename = "standard2.txt";

  Graph *graph = loadGraphUndirected(filename);

  // initialize nodeVisited, currLevelNodes, nextLevelNodes, numCurrLevelNodes,
  // numNextLevelNodes
  int *nodeVisited;
  int *currLevelNodes;
  int *nextLevelNodes;
  int numCurrLevelNodes;
  int *numNextLevelNodes;

  cudaMalloc(&nodeVisited, sizeof(int) * graph->numNodes);
  cudaMalloc(&currLevelNodes, sizeof(int) * graph->numNodes);
  cudaMalloc(&nextLevelNodes, sizeof(int) * graph->numNodes);
  cudaMalloc(&numNextLevelNodes, sizeof(int));

  cudaMemset(nodeVisited, 0, sizeof(int) * graph->numNodes);
  cudaMemset(currLevelNodes, 0, sizeof(int) * graph->numNodes);
  cudaMemset(nextLevelNodes, 0, sizeof(int) * graph->numNodes);
  numCurrLevelNodes = 1;
  cudaMemset(numNextLevelNodes, 0, sizeof(int));

  // set the source node
  cudaMemset(currLevelNodes, 0, sizeof(int));
  int visited = 1;
  cudaMemcpy(nodeVisited, &visited, sizeof(int), cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float kernel_total_time_ms = 0.0f;

  // main loop
  while (numCurrLevelNodes > 0) {
    // start a cuda timer
    cudaEventRecord(start);

    gpu_global_queuing_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(
        graph->nodePtrs, graph->nodeNeighbors, nodeVisited, currLevelNodes,
        nextLevelNodes, numCurrLevelNodes, numNextLevelNodes);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    kernel_total_time_ms += milliseconds;

    // copy numNextLevelNodes to numCurrLevelNodes
    cudaMemcpy(&numCurrLevelNodes, numNextLevelNodes, sizeof(int),
               cudaMemcpyDeviceToHost);
    // reset numNextLevelNodes
    cudaMemset(numNextLevelNodes, 0, sizeof(int));

    // swap currLevelNodes and nextLevelNodes
    int *tmp = currLevelNodes;
    currLevelNodes = nextLevelNodes;
    nextLevelNodes = tmp;
  }

  // print number of visited nodes
  int *nodeVisitedHost = (int *)malloc(sizeof(int) * graph->numNodes);
  cudaMemcpy(nodeVisitedHost, nodeVisited, sizeof(int) * graph->numNodes,
             cudaMemcpyDeviceToHost);
  int numVisitedNodes = 0;
  for (int i = 0; i < graph->numNodes; i++) {
    if (nodeVisitedHost[i] == 1) {
      numVisitedNodes++;
    }
  }
  printf("Number of visited nodes: %d\n", numVisitedNodes);

  // print kernel total time
  printf("Kernel total time: %f ms\n", kernel_total_time_ms);

  return 0;
}