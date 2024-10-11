from typing import List
import pandas as pd, networkx as nx, matplotlib.pyplot as plt,tkinter as tk,numpy as np,math,plotly.graph_objects as go

class Airport: # Clase aeropuerto para almacenar la informacion de cada uno
    def __init__(self, code: str, name: str, city:str, country: str, latitude: float, longitude: float) -> None:
        self.code=code
        self.name=name
        self.city=city
        self.country=country
        self.latitude=latitude
        self.longitude=longitude

class Graph:

    def __init__(self, n: int, directed: bool = False):
        self.n = n
        self.directed = directed
        self.L: List[List[int]] = [[] for _ in range(n)]
        self.c: List[List[List[float]]] = [[None for a in range(n)] for _ in range(n)]

    def add_edge(self, u: int, v: int, distance: float) -> bool:
        if 0 <= u < self.n and 0 <= v < self.n:
            self.L[u].append(v)
            self.L[u].sort()
            self.c[u][v]=distance
            if not self.directed:
                self.L[v].append(u)
                self.L[v].sort()
                self.c[v][u]=distance
            return True
        return False

    def DFS(self, u: int) -> List[bool]:
        visit = [False] * self.n
        self.__DFS_visit(u, visit)

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

    def c_dfs(self,u):
        visit = [False] * self.n
        return self._c_dfs(u,visit)

    def _c_dfs(self,u,visit): #Recorrido DFS sin mostrar los nodos para obtener los nodos visitados
            visit[u] = True
            for v in self.L[u]:
                if not visit[v]:
                    visit = self._c_dfs(v, visit)
            return visit

    def is_connected(self) -> bool: 
        return all(self.__c_dfs(0)) #Retorna True si todos los nodos fueron visitados, si no, retorna False

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

    def components(self) -> List[List[int]]: # Funcion que retorna una lista con listas de vertices por cada componente
        visit=[False]*self.n
        componentes=[]
        for i in range(self.n):
            if visit[i]==False:
                comp=[]
                vstd=self.c_dfs(i)
                for j in range(len(vstd)):
                    if vstd[j]==True:
                        comp.append(j)
                        visit[j]=True
                componentes.append(comp)
        return componentes

    def dijkstra(self,v):
        d=[math.inf]*self.n
        path=[None]*self.n
        visit=[False]*self.n
        d[v]=0
        while not all(visit):
            v=min(d,visit)
            visit[v]=True
            for i in self.L[v]:
                if d[v]+self.c[v][i] < d[i] and not visit[i]:
                    d[i]=d[v]+self.c[v][i]
                    path[i]=v
        return d,path
    
    def floyd_warshall(self):
        d=self.c
        for a in range(len(d)):  #Creacion de la matriz de distancias
            for b in range(len(a)):
                if a!=b and d[a][b]==0:
                    d[a][b]=math.inf
        path=[[]*self.n]*self.n
        for a in range(len(d)): #Creacion de la matriz de caminos
            for b in range(len(a)):
                if a!=b:
                    path[a][b]=b
        for k in range(self.n-1):
            for i in range(self.n-1):
                for j in range(self.n-1):
                    if d[i][k]+d[k][j]<d[i][j]:
                        d[i][j]=d[i][k]+d[k][j]
                        path[i][j]=k
        return d,path

    def bellman_ford(self,v):
        pass

    def kruskal(self):
        pass

    def prim(self,v):
        pass


def AirportCodeToIndex(code: str, airports: List[Airport]): # Funcion para buscar el indice del aeropuerto en la lista de aeropuertos
    for i in range(len(airports)):
        if airports[i].code==code:
            return i
    return False

def distanceFromGeographicCoordinates(latitude: float, longitude: float,latitude2: float, longitude2: float) -> float:
    # Radio de la Tierra en km
    R = 6371.0

    # Convertir las coordenadas de grados a radianes
    lat1_rad = math.radians(latitude)
    lon1_rad = math.radians(longitude)
    lat2_rad = math.radians(latitude2)
    lon2_rad = math.radians(longitude2)

    # Diferencias de latitud y longitud
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

def printGraph():   #Mostrar la representacion del grafo completo
    lons = AirportsDataframe['Longitude'].tolist()  # Lista de longitudes
    lats = AirportsDataframe['Latitude'].tolist()   # Lista de latitudes
    codes = AirportsDataframe['Code'].tolist()      # Lista de códigos
    fig=go.Figure()
    fig.add_trace(go.Scattergeo(lon = lons,lat = lats,text = codes, mode = 'markers',marker=dict(size=8, color='blue'),name="Aeropuertos"))  # Se añade cada aeropuerto con la informacion anterior
    for i in range(len(FlightsDataframe)):
        lat1, lon1 = FlightsDataframe.iloc[i]['Source Airport Latitude'],FlightsDataframe.iloc[i]['Source Airport Longitude']
        lat2, lon2 = FlightsDataframe.iloc[i]['Destination Airport Latitude'],FlightsDataframe.iloc[i]['Destination Airport Longitude']
        #Se añaden las rutas entre aeropuertos del dataframe
        #fig.add_trace(go.Scattergeo(lon = [lon1, lon2],lat = [lat1, lat2],mode = 'lines',line=dict(width=2, color='red'),opacity=0.7,name=f"{FlightsDataframe.iloc[i]['Source Airport Code']} - {FlightsDataframe.iloc[i]['Destination Airport Code']}"))
    fig.update_geos(projection_type="orthographic",showcountries=True,showcoastlines=True,showland=True)
    fig.update_layout(title="Rutas entre Aeropuertos en el Globo Terráqueo",geo=dict(showland=True,landcolor="rgb(243, 243, 243)",oceancolor="rgb(204, 255, 255)",showocean=True))
    fig.show()

def printPath(path: List[int]): #Mostrar un camino entre aeropuertos
    lat=[]
    lon=[]
    labels=[]
    fig=go.Figure()

    for i in range(len(path)):
        labels.append(AirportsDataframe.iloc[i]['Code'])
        lat.append(AirportsDataframe.iloc[i]['Latitude'])
        lon.append(AirportsDataframe.iloc[i]['Longitude'])
        if i!=0:
            lat1, lon1 = AirportsDataframe.iloc[i-1]['Latitude'],AirportsDataframe.iloc[i-1]['Longitude']
            lat2, lon2 = AirportsDataframe.iloc[i]['Latitude'],AirportsDataframe.iloc[i]['Longitude']
            fig.add_trace(go.Scattergeo(lon = [lon1, lon2],lat = [lat1, lat2],mode = 'lines',line=dict(width=2, color='red'),opacity=0.7,name=f"{FlightsDataframe.iloc[AirportCodeToIndex(AirportList[i-1].code,AirportList)]['Source Airport Code']} - {FlightsDataframe.iloc[AirportCodeToIndex(AirportList[i].code,AirportList)]['Destination Airport Code']}"))
    fig.add_trace(go.Scattergeo(lon = lon,lat = lat,text = labels,mode = 'markers',marker=dict(size=8, color='blue'),name="Aeropuertos"))
    fig.update_geos(projection_type="orthographic",showcountries=True,showcoastlines=True,showland=True)
    fig.update_layout(title="Camino minimo entre aeropuertos",geo=dict(showland=True,landcolor="rgb(243, 243, 243)",oceancolor="rgb(204, 255, 255)",showocean=True))
    fig.show()


#Obtencion de dataframes de aeropuertos y rutas entre ellos
FlightsDataframe=pd.read_csv("flights_final.csv")
df1=FlightsDataframe[['Source Airport Code','Source Airport Name','Source Airport City','Source Airport Country','Source Airport Latitude','Source Airport Longitude']]
df2=FlightsDataframe[['Destination Airport Code','Destination Airport Name','Destination Airport City','Destination Airport Country','Destination Airport Latitude','Destination Airport Longitude']]
df1.columns = ['Code', 'Name', 'City', 'Country', 'Latitude', 'Longitude']
df2.columns = ['Code', 'Name', 'City', 'Country', 'Latitude', 'Longitude']
AirportsDataframe=pd.concat([df1, df2], ignore_index=True)
AirportsDataframe.drop_duplicates(subset='Code', inplace=True, ignore_index=True)

#Creacion de lista de Aeropuertos y sus atributos
AirportList=[]
for i in range(len(AirportsDataframe)):
    AirportList.append(Airport(AirportsDataframe.iloc[i]['Code'],AirportsDataframe.iloc[i]['Name'],AirportsDataframe.iloc[i]['City'],AirportsDataframe.iloc[i]['Country'],float(AirportsDataframe.iloc[i]['Latitude']),float(AirportsDataframe.iloc[i]['Longitude'])))

#Creacion del grafo
grafo=Graph(len(AirportsDataframe),False)
for i in range(len(FlightsDataframe)):
    grafo.add_edge(AirportCodeToIndex(FlightsDataframe.iloc[i]['Source Airport Code'],AirportList),AirportCodeToIndex(FlightsDataframe.iloc[i]['Destination Airport Code'],AirportList),distanceFromGeographicCoordinates(float(FlightsDataframe.iloc[i]['Source Airport Latitude']),float(FlightsDataframe.iloc[i]['Source Airport Longitude']),float(FlightsDataframe.iloc[i]['Destination Airport Latitude']),float(FlightsDataframe.iloc[i]['Destination Airport Longitude'])))
