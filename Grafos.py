from typing import List

class Graph:

    def __init__(self, n: int, directed: bool = False):
        self.n = n
        self.directed = directed
        self.L: List[List[int]] = [[] for _ in range(n)]
        self.P: List[List[List[float]]] = [[None for a in range(n)] for _ in range(n)]

    def add_edge(self, u: int, v: int, distance: float) -> bool:
        if 0 <= u < self.n and 0 <= v < self.n:
            self.L[u].append(v)
            self.L[u].sort()
            self.P[u][v]=distance
            if not self.directed:
                self.L[v].append(u)
                self.L[v].sort()
                self.P[v][u]=distance
            return True
        return False

    def DFS(self, u: int) -> List[bool]:
        visit = [False] * self.n
        return self.__DFS_visit(u, visit)

    def __DFS_visit(self, u: int, visit: List[bool]) -> List[bool]:
        visit[u] = True
        print(u, end = ' ')
        for v in self.L[u]:
            if not visit[v]:
                visit = self.__DFS_visit(v, visit)
        return visit

    def BFS(self, u: int) -> None:
        queue = []
        visit = [False] * self.n
        visit[u] = True
        queue.append(u)
        while len(queue) > 0:
            u = queue.pop(0)
            print(u, end = ' ')
            for v in self.L[u]:
                if not visit[v]:
                    visit[v] = True
                    queue.append(v)

    def degree(self, u: int) -> int:
        return len(self.L[u])

    def min_degree(self) -> int:
        return min(len(self.L[i]) for i in range(self.n))

    def max_degree(self) -> int:
        return max(len(self.L[i]) for i in range(self.n))

    def degree_sequence(self) -> List[int]:
        return [len(self.L[i]) for i in range(self.n)]

    def number_of_components(self) -> int:
        visit = [False] * self.n
        count = 0
        for i in range(self.n):
            if not visit[i]:
                visit = self.__DFS_visit(i, visit)
                count += 1
        return count

    def is_connected(self) -> bool:
        return all(self.DFS(1))

    def path(self, u: int, v: int) -> List[int]:
        pass

    def is_eulerian(self) -> bool:
        for i in self.L:
          if len(i)%2!=0:
            return False
        return True

    def is_semieulerian(self) -> bool:
        oddVertex=0
        if not self.is_eulerian():
          for i in self.L:
            if len(i)%2!=0:
              if oddVertex==0:
                oddVertex+=1
              else:
                return True
          return False
        return False

    def is_r_regular(self, r: int) -> bool:
        for i in self.L:
          if len(i)!=r:
            return False
        return True

    def is_complete(self) -> bool:
        if self.is_r_regular(self.n):
            return True
        return False

    def is_acyclic(self) -> bool:
        pass