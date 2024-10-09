import networkx as nx
import plotly.graph_objects as go

# Crear el grafo de aeropuertos
G = nx.Graph()

# Ejemplo de aeropuertos (nombre, latitud, longitud)
aeropuertos = [
    ("JFK", 40.6413, -73.7781),  # Nueva York
    ("LHR", 51.4700, -0.4543),   # Londres
    ("CDG", 49.0097, 2.5479),    # París
    ("HND", 35.5494, 139.7798),  # Tokio
]

# Añadir los nodos al grafo
for aeropuerto in aeropuertos:
    G.add_node(aeropuerto[0], pos=(aeropuerto[2], aeropuerto[1]))

# Añadir conexiones (aristas) entre aeropuertos
G.add_edges_from([("JFK", "LHR"), ("LHR", "CDG"), ("JFK", "CDG"), ("CDG", "HND")])

# Extraer las posiciones de los aeropuertos
pos = nx.get_node_attributes(G, 'pos')

# Listas para almacenar las coordenadas de los nodos (aeropuertos)
lats = [pos[aero][1] for aero in G.nodes()]
lons = [pos[aero][0] for aero in G.nodes()]
labels = list(G.nodes())

# Crear la figura de Plotly para el globo terráqueo
fig = go.Figure()

# Añadir los aeropuertos como puntos en el globo
fig.add_trace(go.Scattergeo(
    lon = lons,
    lat = lats,
    text = labels,
    mode = 'markers',
    marker=dict(size=8, color='blue'),
    name="Aeropuertos"
))

# Añadir las rutas (aristas) como líneas sobre el globo
for edge in G.edges():
    lat1, lon1 = pos[edge[0]]
    lat2, lon2 = pos[edge[1]]
    
    fig.add_trace(go.Scattergeo(
        lon = [lon1, lon2],
        lat = [lat1, lat2],
        mode = 'lines',
        line=dict(width=2, color='red'),
        opacity=0.7,
        name=f"{edge[0]} - {edge[1]}"
    ))

# Configurar la visualización 3D del globo terráqueo
fig.update_geos(
    projection_type="orthographic",  # Proyección en globo
    showcountries=True,             # Mostrar fronteras de los países
    showcoastlines=True,            # Mostrar líneas costeras
    showland=True,                  # Mostrar tierra
)

# Ajustar el layout del gráfico
fig.update_layout(
    title="Rutas entre Aeropuertos en el Globo Terráqueo",
    geo=dict(
        showland=True,
        landcolor="rgb(243, 243, 243)",
        oceancolor="rgb(204, 255, 255)",
        showocean=True
    )
)

# Mostrar la figura interactiva
fig.show()
