#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>

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
 * @return Graph* A host graph structure.
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

  // allocate memory for graph with malloc
  Graph *graph = (Graph *)malloc(sizeof(Graph));
  graph->numNodes = numNodes;

  // allocate memory for nodePtrsHost and nodeNeighborsHost with malloc
  int *nodePtrsHost, *nodeNeighborsHost;
  nodePtrsHost = (int *)malloc(sizeof(int) * (graph->numNodes + 1));
  nodeNeighborsHost = (int *)malloc(sizeof(int) * numEdges);

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

  graph->nodePtrs = nodePtrsHost;
  graph->nodeNeighbors = nodeNeighborsHost;

  // free mySet
  mySet.clear();

  fclose(file);

  return graph;
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

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: %s <graph file>\n", argv[0]);
    return 1;
  }

  const char *filename = argv[1];

  bool mtx = false;
  // if filename ends with .mtx, then it's a Matrix Market file
  if (strlen(filename) > 4 &&
      strcmp(filename + strlen(filename) - 4, ".mtx") == 0) {
    mtx = true;
  }

  Graph *graph = loadGraphUndirected(filename, mtx);

  // traverse the graph
  int *nodeVisited = (int *)calloc(graph->numNodes, sizeof(int));
  int *currLevelNodes = (int *)calloc(graph->numNodes, sizeof(int));
  int *nextLevelNodes = (int *)calloc(graph->numNodes, sizeof(int));

  // initialize nodeVisited
  nodeVisited[0] = 1;

  // initialize numCurrLevelNodes
  int numCurrLevelNodes = 1;

  // initialize numNextLevelNodes
  int numNextLevelNodes = 0;

  int totalNumNodes = 0;

  auto totalTime_ms = 0.0;

  // traverse the graph
  while (numCurrLevelNodes > 0) {
    // traverse the graph sequentially
    auto start = std::chrono::high_resolution_clock::now();
    sequential_cpu_traversal(graph, nodeVisited, currLevelNodes, nextLevelNodes,
                             numCurrLevelNodes, &numNextLevelNodes);
    totalNumNodes += numCurrLevelNodes;
    auto end = std::chrono::high_resolution_clock::now();
    totalTime_ms +=
        std::chrono::duration<double, std::milli>(end - start).count();
    // swap currLevelNodes and nextLevelNodes
    int *tmp = currLevelNodes;
    currLevelNodes = nextLevelNodes;
    nextLevelNodes = tmp;
    // reset numNextLevelNodes
    numCurrLevelNodes = numNextLevelNodes;
    numNextLevelNodes = 0;
  }

  // print number of nodes visited
  int numNodesVisited = 0;
  for (int i = 0; i < graph->numNodes; i++) {
    if (nodeVisited[i] == 1) {
      numNodesVisited++;
    }
  }
  printf("Number of nodes visited: %d\n", numNodesVisited);

  // print total time
  printf("Total time: %f ms\n", totalTime_ms);

  return 0;
}