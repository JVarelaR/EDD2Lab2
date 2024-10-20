from typing import List
import pandas as pd,tkinter as tk,math,plotly.graph_objects as go, heapq
from tkinter import messagebox, scrolledtext

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
        return all(self.c_dfs(0)) #Retorna True si todos los nodos fueron visitados, si no, retorna False

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

    def dijkstra(self,v: int):
        d=[math.inf]*self.n
        path=[None]*self.n
        visit=[False]*self.n
        d[v]=0
        while not all(visit):
            u = None
            min_dist = math.inf  # Encuentra el nodo no visitado con la menor distancia
            for i in range(self.n):
                if not visit[i] and d[i] <= min_dist:
                    min_dist = d[i]
                    u = i
            if u is None:
                break
            
            visit[u] = True
            for i in self.L[u]:
                if d[u] + self.c[u][i] < d[i] and not visit[i]:
                    d[i] = d[u] + self.c[u][i]
                    path[i] = u
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

    def kruskal(self) -> List:
        pass
    
    def prim(self, v: int) -> List:
        q = []
        for u in self.L[v]:
            heapq.heappush(q, (self.c[v][u], [v, u]))
        ver, edg = [v], []
        while len(ver) < self.n and q:
            cost, (vi, vj) = heapq.heappop(q)
            if vj not in ver:
                ver.append(vj)
                edg.append([cost, [vi, vj]])
                for vk in self.L[vj]:
                    heapq.heappush(q, (self.c[vj][vk], [vj, vk]))
        return edg


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
        labels.append("Codigo: "+AirportsDataframe.iloc[path[i]]['Code']+"\nNombre: "+AirportsDataframe.iloc[path[i]]['Name']+"\nCiudad: "+AirportsDataframe.iloc[path[i]]['City']+"\nPais: "+AirportsDataframe.iloc[path[i]]['Country'])
        lat.append(AirportsDataframe.iloc[path[i]]['Latitude'])
        lon.append(AirportsDataframe.iloc[path[i]]['Longitude'])
        if i!=0:
            lat1, lon1 = AirportsDataframe.iloc[path[i-1]]['Latitude'],AirportsDataframe.iloc[path[i-1]]['Longitude']
            lat2, lon2 = AirportsDataframe.iloc[path[i]]['Latitude'],AirportsDataframe.iloc[path[i]]['Longitude']
            fig.add_trace(go.Scattergeo(lon = [lon1, lon2],lat = [lat1, lat2],mode = 'lines',line=dict(width=2, color='red'),opacity=0.7, name=f"{AirportList[path[i-1]].code} - {AirportList[path[i]].code}"))
    fig.add_trace(go.Scattergeo(lon = lon,lat = lat,text = labels,mode = 'markers',marker=dict(size=8, color='blue'),name="Aeropuertos"))
    fig.update_geos(projection_type="orthographic",showcountries=True,showcoastlines=True,showland=True)
    fig.update_layout(title="Camino minimo entre aeropuertos",geo=dict(showland=True,landcolor="rgb(243, 243, 243)",oceancolor="rgb(204, 255, 255)",showocean=True))
    fig.show()

def printAirports(path: List[int]): #Mostrar aeropuertos especificos
    lat=[]
    lon=[]
    labels=[]
    fig=go.Figure()
    for i in range(len(path)):
        labels.append("Codigo: "+AirportsDataframe.iloc[i]['Code']+"\nNombre: "+AirportsDataframe.iloc[i]['Name']+"\nCiudad: "+AirportsDataframe.iloc[i]['City']+"\nPais: "+AirportsDataframe.iloc[i]['Country']+"\nLatitud: "+str(AirportsDataframe.iloc[i]['Latitude'])+"\nLongitud: "+str(AirportsDataframe.iloc[i]['Longitude']))
        lat.append(AirportsDataframe.iloc[i]['Latitude'])
        lon.append(AirportsDataframe.iloc[i]['Longitude'])
    fig.add_trace(go.Scattergeo(lon = lon,lat = lat,text = labels,mode = 'markers',marker=dict(size=8, color='blue'),name="Aeropuertos"))
    fig.update_geos(projection_type="orthographic",showcountries=True,showcoastlines=True,showland=True)
    fig.update_layout(title="Aeropuertos",geo=dict(showland=True,landcolor="rgb(243, 243, 243)",oceancolor="rgb(204, 255, 255)",showocean=True))
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


def show_root(frame):
    frame.withdraw() 
    root.deiconify()
def show_functionsWindow(frame):
    frame.withdraw() 
    functionsWindow.deiconify()
def show_selectAirportWindow(frame):
    frame.withdraw() 
    selectAirportWindow.deiconify()
def show_airportInfoWindow(code:str):
    airportIndex=AirportCodeToIndex(code,AirportList)
    if airportIndex==False:
        messagebox.showerror(title="Error", message="No se encontró un aeropuerto con ese codigo")
    else:
        # Cuarta ventana (Informacion del aeropuerto)
        airportInfoWindow = tk.Toplevel()
        airportInfoWindow.title("Caminos Minimos")
        airportInfoWindow.geometry("1300x500")
        airportInfoWindow.resizable(False,False) 
        tk.Label(airportInfoWindow, text="Caminos minimos más largos:", font=("Arial", 14)).place(rely=0.15,relx=0.5, anchor="center")
        airportsInfoText7=scrolledtext.ScrolledText(airportInfoWindow, width=150,height=11)
        airportsInfoText7.config(state="normal")
        airportsInfoText7.insert(tk.INSERT,"Codigo:\tNombre:\t\t\t\tCiudad:\t\t\tPaís:\t\t\t\tLatitud:\t\tLongitud:\t\tDistancia:\n")
        
        #----------------Agregar la informacion de los caminos mas largos
        d,paths=grafo.dijkstra(airportIndex)
        vertices=list(range(len(d)))
        dist,vertex=zip(*sorted(zip(d,vertices),reverse=True))
        dist=list(dist)
        vertex=list(vertex)
        i=0
        c=0
        while c<10:
            if dist[i]!=math.inf: #Se escogen los aeropuertos sin distancias infinitas
                airportsInfoText7.insert(tk.INSERT,f"{AirportList[vertex[i]].code}\t{AirportList[vertex[i]].name[0:15]}\t\t\t\t{AirportList[vertex[i]].city}\t\t\t{AirportList[vertex[i]].country}\t\t\t\t{AirportList[vertex[i]].latitude}\t\t{AirportList[vertex[i]].longitude}\t\t{dist[i]}\n")
                c+=1
            i+=1
        airportsInfoText7.config(state="disabled")

        airportsInfoText7.place(relx=0.5,rely=0.39, anchor="center")
        tk.Label(airportInfoWindow, text="Inserte el codigo de otro aeropuerto para buscar el camino minimo entre ellos:", font=("Arial", 11)).place(rely=0.7,relx=0.5, anchor="center")
        secondAirportEntry4 = tk.Entry(airportInfoWindow, width=30)
        secondAirportEntry4.place(rely=0.78,relx=0.5, anchor="center")
        backButton4 = tk.Button(airportInfoWindow, text="Volver", command=lambda: airportInfoWindow.withdraw())
        backButton4.place(x=10,y=10)
        minimumPathButton4 = tk.Button(airportInfoWindow, text="Buscar camino minimo", command=lambda: show_minimum_path(code,secondAirportEntry4.get()))
        minimumPathButton4.place(rely=0.9,relx=0.5, anchor="center")

        messagebox.showinfo(title="Información del aeropuerto", message=f"Codigo: {AirportList[airportIndex].code}\nNombre: {AirportList[airportIndex].name}\nCiudad: {AirportList[airportIndex].city}\nPaís: {AirportList[airportIndex].country}\nLatitud: {AirportList[airportIndex].latitude}\nLongitud: {AirportList[airportIndex].longitude}")

def show_connection():
    components=grafo.components()
    comNodes=""
    if grafo.is_connected():
        connected="El grafo es conexo"
    else:
        connected="El grafo no es conexo"
    for i in range(len(components)):
        comNodes+=f"El componente {i+1} tiene {len(components[i])} vértices\n"
    messagebox.showinfo(title="Información del grafo", message=f"{connected}\nTiene {len(components)} componentes\n{comNodes}")
def show_minimum_expansion():
    components=grafo.components()
    weights=""
    for i in range(len(components)):
        weight=0
        for j in grafo.prim(components[i][0]):
            weight+=j[0]
        weights+=f"El componente {i+1} tiene un arbol de expansión con peso {weight}\n"
    messagebox.showinfo(title="Arboles de expansión minima", message=f"El arbol tiene {len(components)} componentes\n{weights}")
def show_minimum_path(code1: str, code2: str):
    airport1=AirportCodeToIndex(code1,AirportList)
    airport2=AirportCodeToIndex(code2,AirportList)
    if airport1 is False or airport2 is False:
        messagebox.showerror(title="Error", message="No se encontró un aeropuerto con ese código")
        return
    
    if airport1==airport2:
        printPath([airport1]) #El aeropuerto inicial y final son el mismo
        return
    
    d,paths=grafo.dijkstra(airport1)
    if d[airport2]==math.inf:
        messagebox.showerror(title="Error", message="Los aeropuertos no estan conectados")
        return

    path=[]
    v=airport2
    while v is not None and v!=airport1:
        path.append(v)
        v=paths[v]

    if v is None:
        # Esto significa que no se pudo construir un camino adecuado
        messagebox.showerror(title="Error", message="No se pudo encontrar un camino válido")
        return

    path.append(airport1)
    path=path[::-1]
    printPath(path)

#Interfaz Grafica

# Primera ventana (Inicio)
root = tk.Tk()
root.title("Inicio")
root.geometry("500x300")
root.resizable(False,False) 
label1 = tk.Label(root, text="Rutas entre aeropuertos", font=("Arial", 14))
label1.place(rely=0.25,relx=0.5, anchor="center")
continue_button = tk.Button(root, text="Continuar", command=lambda: show_functionsWindow(root))
continue_button.place(rely=0.7,relx=0.5, anchor="center")

# Segunda ventana (Funciones)
functionsWindow = tk.Toplevel()
functionsWindow.protocol('WM_DELETE_WINDOW', lambda: root.destroy())
functionsWindow.title("Funciones")
functionsWindow.geometry("600x300")
functionsWindow.resizable(False,False) 
functionsWindow.withdraw()  # Esconde la ventana al inicio
label2 = tk.Label(functionsWindow, text="Funciones", font=("Arial", 14))
label2.place(rely=0.15,relx=0.5, anchor="center")
backButton2 = tk.Button(functionsWindow, text="Volver", command=lambda: show_root(functionsWindow))
backButton2.place(x=10,y=10)
visualizationButton2 = tk.Button(functionsWindow, text="Visualizar Mapa", command=lambda: printGraph())
visualizationButton2.place(rely=0.4,relx=0.5, anchor="center")
connectionButton2 = tk.Button(functionsWindow, text="Conexidad", command=lambda: show_connection())
connectionButton2.place(rely=0.55,relx=0.5, anchor="center")
minimumExpansionButton2 = tk.Button(functionsWindow, text="Arboles de expansion minima", command=lambda: show_minimum_expansion())
minimumExpansionButton2.place(rely=0.7,relx=0.5, anchor="center")
selectAirportButton2 = tk.Button(functionsWindow, text="Selección de aeropuerto", command=lambda: show_selectAirportWindow(functionsWindow))
selectAirportButton2.place(rely=0.85,relx=0.5, anchor="center")

# Tercera ventana (Seleccion de un aeropuerto)
selectAirportWindow = tk.Toplevel()
selectAirportWindow.title("Selección de un aeropuerto")
selectAirportWindow.geometry("600x300")
selectAirportWindow.resizable(False,False)
selectAirportWindow.withdraw()
selectAirportWindow.protocol('WM_DELETE_WINDOW', lambda: root.destroy())
tk.Label(selectAirportWindow, text="Selección de un aeropuerto", font=("Arial", 14)).pack(pady=20)
tk.Label(selectAirportWindow, text="Ingrese el codigo del aeropuerto a seleccionar:", font=("Arial", 12)).pack(pady=20)
airportCodeEntry3 = tk.Entry(selectAirportWindow, width=30)
airportCodeEntry3.pack(pady=10)
backButton3 = tk.Button(selectAirportWindow, text="Volver", command=lambda: show_functionsWindow(selectAirportWindow))
backButton3.place(x=10,y=10)
searchButton3 = tk.Button(selectAirportWindow, text="Buscar", command=lambda: show_airportInfoWindow(airportCodeEntry3.get()))
searchButton3.pack(pady=20)



# Inicia el bucle principal de la aplicación
root.mainloop()