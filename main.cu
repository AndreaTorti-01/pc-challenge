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

// Define a structure for the current level nodes
typedef struct {
  int *nodes;
  int numNodes;
} CurrentLevel;

// Define a structure for visited nodes
typedef struct {
  int *nodes;
  int numNodes;
} VisitedNodes;

// Define a structure for the next level nodes
typedef struct {
  int *nodes;
  int numNodes;
} NextLevel;

/**
 * @brief Loads a graph from a file.
 *
 * @param filename The name of the file containing the graph data.
 * @return Graph* A pointer to the loaded graph.
 */
Graph *loadOrderedGraphDirected(const char *filename) {
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    perror("Error opening file");
    return NULL;
  }

  // allocate memory for the graph with cudaMalloc
  Graph *graph;
  cudaError_t err = cudaMallocManaged(&graph, sizeof(Graph));
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
  }

  // set number of nodes reading it from the file
  fscanf(file, "%d", &graph->numNodes);

  // scan total number of edges reading it from the file
  int numEdges;
  fscanf(file, "%d", &numEdges);

  // allocate memory for nodePtrs and nodeNeighbors with cudaMalloc
  cudaMallocManaged(&graph->nodePtrs, sizeof(int) * (graph->numNodes + 1));
  cudaMallocManaged(&graph->nodeNeighbors, sizeof(int) * numEdges);

  int sourceNode = 0;
  int edgeIdx = 0;
  int lastSourceNode = 0;
  graph->nodePtrs[0] = 0;
  while (fscanf(file, "%d", &sourceNode) != EOF) {
    // fill nodePtrs array
    if (sourceNode != lastSourceNode) {
      graph->nodePtrs[lastSourceNode + 1] = edgeIdx;
      lastSourceNode = sourceNode;
    }

    // fill nodeNeighbors array
    fscanf(file, "%d", &graph->nodeNeighbors[edgeIdx]);
    edgeIdx++;
    // fill last element of nodePtrs array
    graph->nodePtrs[lastSourceNode + 1] = edgeIdx;
  }

  fclose(file);

  return graph;
}

Graph *loadOrderedGraphUndirected(const char *filename) {
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    perror("Error opening file");
    return NULL;
  }

  int numNodes = 0;
  fscanf(file, "%d", &numNodes);
  bool** matrix_graph = (bool**)calloc(numNodes, sizeof(bool*));

  // start by filling the matrix with zeros
  for (int i = 0; i < numNodes; i++) {
    matrix_graph[i] = (bool*)calloc(numNodes, sizeof(bool));
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
    for (int j = i; j < numNodes; j++) {
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
  cudaMallocManaged(&graph->nodePtrs, sizeof(int) * (graph->numNodes + 1));
  cudaMallocManaged(&graph->nodeNeighbors, sizeof(int) * numEdges);

  // fill nodePtrs and nodeNeighbors arrays
  int edgeIdx = 0;
  graph->nodePtrs[0] = 0;
  for (int i = 0; i < numNodes; i++) {
    for (int j = i; j < numNodes; j++) {
      if (matrix_graph[i][j]) {
        graph->nodeNeighbors[edgeIdx] = j;
        edgeIdx++;
      }
    }
    graph->nodePtrs[i + 1] = edgeIdx;
  }

  // free memory
  for (int i = 0; i < numNodes; i++) {
    free(matrix_graph[i]);
  }

  free(matrix_graph);

  fclose(file);

  return graph;
}

Graph *loadMTXGraph(const char *filename) {
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    return NULL;
  }

  // Read the header
  int numNodes, numEdges;
  while (1) {
    char line[1024];
    fgets(line, 1024, file);
    if (line[0] != '%') {
      sscanf(line, "%d %d %d", &numNodes, &numNodes, &numEdges);
      break;
    }
  }

  // Allocate memory for the graph structure on the GPU
  Graph *graph;
  cudaMallocManaged(&graph, sizeof(Graph));
  graph->numNodes = numNodes;

  cudaMallocManaged(&graph->nodePtrs, (numNodes + 1) * sizeof(int));

  cudaMallocManaged(&graph->nodeNeighbors, numEdges * sizeof(int));

  // Read the data and fill the structure
  int currentRow = -1;
  int edgeCount = 0;
  graph->nodePtrs[0] = 0;
  while (!feof(file)) {
    int row, col;
    fscanf(file, "%d %d", &row, &col);

    // MTX format is 1-based, so we need to convert to 0-based
    row--;
    col--;

    while (currentRow < row) {
      currentRow++;
      graph->nodePtrs[currentRow + 1] = edgeCount;
    }
    graph->nodeNeighbors[edgeCount] = col;
    edgeCount++;
  }

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

// Block queuing stub
__global__ void gpu_block_queuing_kernel(int *nodePtrs, int *nodeNeighbors,
                                         int *nodeVisited, int *currLevelNodes,
                                         int *nextLevelNodes,
                                         const int numCurrLevelNodes,
                                         int *numNextLevelNodes) {
  //@@ Insert Block Queuing Code Here
  // Initialize shared memory queue
  // Loop over all nodes in the current level
  // Loop over all neighbors of the node
  // If the neighbor hasn't been visited yet
  // Add it to the block queue
  // If full, add it to the global queue
  // Allocate space for block queue to go into global queue
  // Store block queue in global queue
}

// Host function for global queuing invocation
void gpu_global_queuing(int *nodePtrs, int *nodeNeighbors, int *nodeVisited,
                        int *currLevelNodes, int *nextLevelNodes,
                        const int numCurrLevelNodes, int *numNextLevelNodes) {
  gpu_global_queuing_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(
      nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
      numCurrLevelNodes, numNextLevelNodes);
  cudaDeviceSynchronize();
}

// Host function for block queuing invocation
void gpu_block_queuing(int *nodePtrs, int *nodeNeighbors, int *nodeVisited,
                       int *currLevelNodes, int *nextLevelNodes,
                       int numCurrLevelNodes, int *numNextLevelNodes) {
  const int numBlocks = 45;
  gpu_block_queuing_kernel<<<numBlocks, BLOCK_SIZE>>>(
      nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
      numCurrLevelNodes, numNextLevelNodes);
}

void sequential_cpu_traversal(Graph *graph, int *nodeVisited,
                              int *currLevelNodes, int *nextLevelNodes,
                              int numCurrLevelNodes, int *numNextLevelNodes) {
  // Iterate over the nodes in the current level.
  for (int i = 0; i < numCurrLevelNodes; i++) {
    // Get the node at the current index.
    int node = currLevelNodes[i];
    // Iterate over the neighbors of the current node.
    int firstNeighbor = graph->nodePtrs[node];
    int lastNeighbor = graph->nodePtrs[node + 1];
    for (int j = firstNeighbor; j < lastNeighbor; j++) {
      // Get the neighbor at the current index.
      int neighbor = graph->nodeNeighbors[j];
      // If the neighbor has not been visited yet.
      if (nodeVisited[neighbor] == 0) {
        // Mark the neighbor as visited.
        nodeVisited[neighbor] = 1;
        // Add the neighbor to the list of nodes to visit in the next level.
        nextLevelNodes[(*numNextLevelNodes)++] = neighbor;
      }
    }
  }
}

int main() {
  const char *filename = "standard2.txt";

  Graph *graph = loadOrderedGraphUndirected(filename);

  // initialize nodeVisited, currLevelNodes, nextLevelNodes, numCurrLevelNodes,
  // numNextLevelNodes
  int *nodeVisited;
  int *currLevelNodes;
  int *nextLevelNodes;
  int numCurrLevelNodes;
  int *numNextLevelNodes;

  cudaMallocManaged(&nodeVisited, sizeof(int) * graph->numNodes);
  cudaMallocManaged(&currLevelNodes, sizeof(int) * graph->numNodes);
  cudaMallocManaged(&nextLevelNodes, sizeof(int) * graph->numNodes);
  cudaMallocManaged(&numNextLevelNodes, sizeof(int));

  cudaMemset(nodeVisited, 0, sizeof(int) * graph->numNodes);
  cudaMemset(currLevelNodes, 0, sizeof(int) * graph->numNodes);
  cudaMemset(nextLevelNodes, 0, sizeof(int) * graph->numNodes);
  cudaMemset(numNextLevelNodes, 0, sizeof(int));

  numCurrLevelNodes = 1;
  *numNextLevelNodes = 0;

  // set the source node
  currLevelNodes[0] = 0;
  nodeVisited[0] = 1;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  auto kernel_total_time_ms = 0;

  // main loop
  while (numCurrLevelNodes > 0) {
    // start a cuda timer
    cudaEventRecord(start);

    gpu_global_queuing_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(
        graph->nodePtrs, graph->nodeNeighbors, nodeVisited, currLevelNodes,
        nextLevelNodes, numCurrLevelNodes, numNextLevelNodes);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    kernel_total_time_ms += milliseconds;

    // copy numNextLevelNodes to numCurrLevelNodes
    numCurrLevelNodes = *numNextLevelNodes;
    // reset numNextLevelNodes
    *numNextLevelNodes = 0;

    // swap currLevelNodes and nextLevelNodes
    int *tmp = currLevelNodes;
    currLevelNodes = nextLevelNodes;
    nextLevelNodes = tmp;
  }

  // print number of visited nodes
  int numVisitedNodes = 0;
  for (int i = 0; i < graph->numNodes; i++) {
    if (nodeVisited[i] == 1) {
      numVisitedNodes++;
    }
  }
  printf("Number of visited nodes: %d\n", numVisitedNodes);

  // print kernel total time
  printf("Kernel total time: %d ms\n", kernel_total_time_ms);

  return 0;
}