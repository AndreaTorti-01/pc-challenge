#include <cstdio>
#include <set>

#define BLOCK_QUEUE_CAP 4096

// Graph structure
typedef struct {
  int numNodes;  // number of nodes
  int *nodePtrs; // each int represents a position in the nodeNeighbors array:
                 // it's the first neighbor of node i (the last one is
                 // nodePtrs[i+1]-1)
  int *nodeNeighbors; // each int represents a neighbor of the node i as above
                      // explained
} Graph;

struct TwoInts {
  int src;
  int dest;

  // Define operator< for sorting
  bool operator<(const TwoInts &other) const {
    if (src != other.src) {
      return src < other.src;
    }
    return dest < other.dest;
  }
};

/**
 * @brief Loads a graph from a file and interprets it as an undirected graph.
 * first line is the number of nodes and the number of edges, then each line is
 * an edge.
 *
 * @param filename The name of the file to load the graph from.
 * @param mtx If true, the file is interpreted as a Matrix Market file (1-based
 * indices).
 * @return Graph* A GPU pointer to the graph structure.
 */
Graph *loadGraphUndirected(const char *filename, bool mtx) {
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    perror("Error opening file");
    return NULL;
  }

  int numNodes = 0;
  fscanf(file, "%d", &numNodes);
  // scan total number of edges reading it from the file and dump it
  int numEdges;
  fscanf(file, "%d", &numEdges);

  std::set<TwoInts> mySet;

  // read the file line by line and fill mySet
  int src, dest;
  if (!mtx) {
    while (fscanf(file, "%d %d", &src, &dest) != EOF) {
      mySet.insert({src, dest});
      mySet.insert({dest, src}); // add the reverse edge
    }
  } else {
    while (fscanf(file, "%d %d", &dest, &src) != EOF) {
      mySet.insert({src - 1, dest - 1});
      mySet.insert({dest - 1, src - 1}); // add the reverse edge
    }
  }

  // count the edges
  numEdges = mySet.size();

  // allocate memory for the graph with cudaMallocManaged
  Graph *graph;
  cudaMallocManaged(&graph, sizeof(Graph));

  // set number of nodes
  graph->numNodes = numNodes;

  // allocate memory for nodePtrs and nodeNeighbors with cudaMallocManaged
  cudaMalloc(&graph->nodePtrs, sizeof(int) * (graph->numNodes + 1));
  cudaMalloc(&graph->nodeNeighbors, sizeof(int) * numEdges);

  // allocate memory for nodePtrsHost and nodeNeighborsHost with cudaMallocHost
  int *nodePtrsHost, *nodeNeighborsHost;
  cudaMallocHost(&nodePtrsHost, sizeof(int) * (graph->numNodes + 1));
  cudaMallocHost(&nodeNeighborsHost, sizeof(int) * numEdges);

  // fill nodePtrs and nodeNeighbors arrays
  int currNode = 0;
  int currEdge = 0;
  for (auto it = mySet.begin(); it != mySet.end(); it++) {
    if (it->src == currNode) {
      nodeNeighborsHost[currEdge] = it->dest;
      currEdge++;
    } else {
      for (int i = currNode + 1; i <= it->src; i++) {
        nodePtrsHost[i] = currEdge;
      }
      nodeNeighborsHost[currEdge] = it->dest;
      currEdge++;
      currNode = it->src; // update currNode to the current source node
    }
  }

  // free mySet
  mySet.clear();

  // copy nodePtrs and nodeNeighbors arrays to device
  cudaMemcpy(graph->nodePtrs, nodePtrsHost, sizeof(int) * (graph->numNodes),
             cudaMemcpyHostToDevice);
  cudaMemcpy(graph->nodeNeighbors, nodeNeighborsHost, sizeof(int) * numEdges,
             cudaMemcpyHostToDevice);

  fclose(file);

  return graph;
}

// Global queuing kernel
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
    int firstNeighbor = nodePtrs[node];
    int lastNeighbor = nodePtrs[node + 1];
    for (int j = firstNeighbor; j < lastNeighbor; j++) {
      // Get the neighbor at the current index.
      int neighbor = nodeNeighbors[j];
      // If the neighbor has not been visited yet.
      if (atomicCAS(&nodeVisited[neighbor], 0, 1) == 0) {
        // Add the neighbor to the list of nodes to visit in the next level.
        // The atomicAdd function ensures that this operation is thread-safe.
        nextLevelNodes[atomicAdd(numNextLevelNodes, 1)] = neighbor;
      }
    }
  }
}

// Block queuing kernel
__global__ void gpu_block_queuing_kernel(int *nodePtrs, int *nodeNeighbors,
                                         int *nodeVisited, int *currLevelNodes,
                                         int *nextLevelNodes,
                                         const int numCurrLevelNodes,
                                         int *numNextLevelNodes) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  // shared_queue[0] is the size of the queue
  __shared__ int shared_mem[BLOCK_QUEUE_CAP + 1];

  int *shared_queue_size = &shared_mem[0];
  int *shared_queue = &shared_mem[1];

  if (threadIdx.x == 0) {
    shared_queue_size[0] = 0;
  }
  __syncthreads();

  // Iterate over the nodes in the current level. The loop stride is the total
  // number of threads.
  for (int i = idx; i < numCurrLevelNodes; i += stride) {
    // Get the node at the current index.
    int node = currLevelNodes[i];
    int firstNeighbor = nodePtrs[node];
    int lastNeighbor = nodePtrs[node + 1];
    for (int j = firstNeighbor; j < lastNeighbor; j++) {
      // Get the neighbor at the current index.
      int neighbor = nodeNeighbors[j];
      // If the neighbor has not been visited yet.
      if (atomicCAS(&nodeVisited[neighbor], 0, 1) == 0) {
        int index =
            atomicAdd_block(shared_queue_size, 1); // avaiable since arch=sm_70
        // if there is space in the shared queue
        if (index < BLOCK_QUEUE_CAP) {
          // add the neighbor to the shared queue
          shared_queue[index] = neighbor;
        } else {
          // if there is no space in the shared queue, add the neighbor to the
          // global queue
          nextLevelNodes[atomicAdd(numNextLevelNodes, 1)] = neighbor;
        }
      }
    }
  }
  __syncthreads();

  // copy the shared queue to the global queue
  if (*shared_queue_size > BLOCK_QUEUE_CAP)
    *shared_queue_size = BLOCK_QUEUE_CAP;

  __shared__ int global_queue_index;
  if (threadIdx.x == 0) {
    global_queue_index = atomicAdd(numNextLevelNodes, *shared_queue_size);
  }

  __syncthreads();

  for (int i = threadIdx.x; i < *shared_queue_size; i += blockDim.x) {
    nextLevelNodes[global_queue_index + i] = shared_queue[i];
  }
}

int main(int argc, char **argv) {
  if (argc < 4) {
    printf("Usage: %s <graph file> <num blocks> <block size>\n", argv[0]);
    return 1;
  }

  const char *filename = argv[1];
  const int NUM_BLOCKS = atoi(argv[2]);
  const int BLOCK_SIZE = atoi(argv[3]);

  bool mtx = false;
  // if filename ends with .mtx, then it's a Matrix Market file
  if (strlen(filename) > 4 &&
      strcmp(filename + strlen(filename) - 4, ".mtx") == 0) {
    mtx = true;
  }

  Graph *graph = loadGraphUndirected(filename, mtx);

  int *nodeVisited;
  int *currLevelNodes;
  int *nextLevelNodes;
  int numCurrLevelNodes;
  int *numNextLevelNodes;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMalloc(&nodeVisited, sizeof(int) * graph->numNodes);
  cudaMalloc(&currLevelNodes, sizeof(int) * graph->numNodes);
  cudaMalloc(&nextLevelNodes, sizeof(int) * graph->numNodes);
  cudaMalloc(&numNextLevelNodes, sizeof(int));

  gpu_global_queuing_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(
      graph->nodePtrs, graph->nodeNeighbors, nodeVisited, currLevelNodes,
      nextLevelNodes, numCurrLevelNodes,
      numNextLevelNodes); // empty call to have consistent timing

  // reset
  cudaMemset(nodeVisited, 0, sizeof(int) * graph->numNodes);
  cudaMemset(currLevelNodes, 0, sizeof(int) * graph->numNodes);
  cudaMemset(nextLevelNodes, 0, sizeof(int) * graph->numNodes);
  numCurrLevelNodes = 1;
  cudaMemset(numNextLevelNodes, 0, sizeof(int));
  cudaMemset(currLevelNodes, 0, sizeof(int));
  int visited = 1;
  cudaMemcpy(nodeVisited, &visited, sizeof(int), cudaMemcpyHostToDevice);
  float kernel_total_time_ms = 0.0f;

  // main loop (global queuing)
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
  int *nodeVisitedHost;
  cudaMallocHost(&nodeVisitedHost, sizeof(int) * graph->numNodes);
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
  printf("Global queuing kernel total time: %f ms\n", kernel_total_time_ms);

  // reset
  cudaMemset(nodeVisited, 0, sizeof(int) * graph->numNodes);
  cudaMemset(currLevelNodes, 0, sizeof(int) * graph->numNodes);
  cudaMemset(nextLevelNodes, 0, sizeof(int) * graph->numNodes);
  numCurrLevelNodes = 1;
  cudaMemset(numNextLevelNodes, 0, sizeof(int));
  cudaMemset(currLevelNodes, 0, sizeof(int));
  visited = 1;
  cudaMemcpy(nodeVisited, &visited, sizeof(int), cudaMemcpyHostToDevice);
  kernel_total_time_ms = 0.0f;

  // main loop (block queuing)
  while (numCurrLevelNodes > 0) {
    // start a cuda timer
    cudaEventRecord(start);

    gpu_block_queuing_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(
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
  cudaMallocHost(&nodeVisitedHost, sizeof(int) * graph->numNodes);
  cudaMemcpy(nodeVisitedHost, nodeVisited, sizeof(int) * graph->numNodes,
             cudaMemcpyDeviceToHost);
  numVisitedNodes = 0;
  for (int i = 0; i < graph->numNodes; i++) {
    if (nodeVisitedHost[i] == 1) {
      numVisitedNodes++;
    }
  }
  printf("Number of visited nodes: %d\n", numVisitedNodes);

  // print kernel total time
  printf("Block queuing kernel total time: %f ms\n", kernel_total_time_ms);

  return 0;
}