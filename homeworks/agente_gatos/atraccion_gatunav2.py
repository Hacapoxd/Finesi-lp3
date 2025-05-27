# Importación de librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import random
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)  # Para reproducibilidad
random.seed(42)

print("✓ Librerías cargadas exitosamente")
print("✓ Configuración de visualización establecida")

# Parámetros del modelo de atracción de gatos
class ParametrosGatos:
    def __init__(self):
        self.tamaño_grilla = 40        # Tamaño de la grilla (NxN)
        self.densidad = 0.3           # Proporción de celdas ocupadas (0-1)
        self.proporcion_hembras = 0.02  # Proporción de hembras en celo vs machos
        self.max_machos_por_hembra = 40 # Máximo número de machos alrededor de una hembra para terminar
        self.max_iteraciones = 100     # Máximo número de iteraciones
        self.radio_atraccion = 10      # Radio de atracción de las hembras en celo
        self.radio_vecindario = 4   # Radio para contar vecinos cercanos

    def mostrar_parametros(self):
        """Muestra los parámetros actuales del modelo"""
        print("📋 PARÁMETROS DEL MODELO DE GATOS")
        print(f"   • Tamaño de grilla: {self.tamaño_grilla}x{self.tamaño_grilla}")
        print(f"   • Densidad poblacional: {self.densidad*100:.1f}%")
        print(f"   • Proporción Hembras en celo: {self.proporcion_hembras*100:.1f}%")
        print(f"   • Proporción Machos: {(1-self.proporcion_hembras)*100:.1f}%")
        print(f"   • Máx. machos por hembra: {self.max_machos_por_hembra}")
        print(f"   • Radio de atracción: {self.radio_atraccion}")
        print(f"   • Máx. iteraciones: {self.max_iteraciones}")

# Crear instancia de parámetros
params = ParametrosGatos()
params.mostrar_parametros()

class ModeloAtraccionGatos:
    def __init__(self, parametros):
        self.params = parametros
        self.grilla = None
        self.historial_agrupacion = []
        self.historial_distancia_promedio = []
        self.iteracion_actual = 0
        self.convergencia_alcanzada = False
        self.hembras_saturadas = []

    def inicializar_poblacion(self):
        """Inicializa la grilla con gatos distribuidos aleatoriamente"""
        N = self.params.tamaño_grilla

        # Crear grilla vacía (0 = vacío, 1 = Macho, 2 = Hembra en celo)
        self.grilla = np.zeros((N, N), dtype=int)

        # Calcular población total
        total_celdas = N * N
        poblacion_total = int(total_celdas * self.params.densidad)

        # Distribuir grupos
        hembras_size = int(poblacion_total * self.params.proporcion_hembras)
        machos_size = poblacion_total - hembras_size

        # Crear lista de tipos de agentes
        tipos_agentes = [1] * machos_size + [2] * hembras_size
        random.shuffle(tipos_agentes)

        # Obtener posiciones disponibles y mezclarlas
        posiciones_disponibles = [(i, j) for i in range(N) for j in range(N)]
        random.shuffle(posiciones_disponibles)

        # Asignar agentes a posiciones
        for idx, tipo in enumerate(tipos_agentes):
            i, j = posiciones_disponibles[idx]
            self.grilla[i, j] = tipo

        print(f"✓ Población inicializada: {machos_size} machos, {hembras_size} hembras en celo")

    def obtener_vecinos(self, x, y, radio=None):
        """Obtiene las coordenadas de los vecinos de una celda dentro del radio especificado"""
        N = self.params.tamaño_grilla
        if radio is None:
            radio = self.params.radio_vecindario
        
        vecinos = []
        for dx in range(-radio, radio + 1):
            for dy in range(-radio, radio + 1):
                if dx == 0 and dy == 0:  # Excluir la celda central
                    continue

                nx, ny = x + dx, y + dy

                # Verificar límites
                if 0 <= nx < N and 0 <= ny < N:
                    vecinos.append((nx, ny))

        return vecinos

    def encontrar_hembra_mas_cercana(self, x, y):
        """Encuentra la hembra en celo más cercana a un macho"""
        N = self.params.tamaño_grilla
        min_distancia = float('inf')
        hembra_cercana = None

        for i in range(N):
            for j in range(N):
                if self.grilla[i, j] == 2:  # Hembra en celo
                    distancia = abs(x - i) + abs(y - j)  # Distancia Manhattan
                    if distancia < min_distancia:
                        min_distancia = distancia
                        hembra_cercana = (i, j)

        return hembra_cercana, min_distancia

    def calcular_direccion_hacia_hembra(self, x_macho, y_macho, x_hembra, y_hembra):
        """Calcula la mejor dirección para que el macho se acerque a la hembra"""
        dx = x_hembra - x_macho
        dy = y_hembra - y_macho

        # Normalizar direcciones (solo movimientos de una celda)
        if dx > 0:
            dx = 1
        elif dx < 0:
            dx = -1
        else:
            dx = 0

        if dy > 0:
            dy = 1
        elif dy < 0:
            dy = -1
        else:
            dy = 0

        return dx, dy

    def contar_machos_alrededor_hembra(self, x_hembra, y_hembra):
        """Cuenta cuántos machos hay alrededor de una hembra"""
        vecinos = self.obtener_vecinos(x_hembra, y_hembra, self.params.radio_vecindario)
        machos_cercanos = 0
        
        for vx, vy in vecinos:
            if self.grilla[vx, vy] == 1:  # Es un macho
                machos_cercanos += 1
                
        return machos_cercanos

    def verificar_convergencia(self):
        """Verifica si alguna hembra tiene 6 o más machos alrededor"""
        N = self.params.tamaño_grilla
        hembras_saturadas = 0

        for i in range(N):
            for j in range(N):
                if self.grilla[i, j] == 2:  # Hembra en celo
                    machos_alrededor = self.contar_machos_alrededor_hembra(i, j)
                    if machos_alrededor >= self.params.max_machos_por_hembra:
                        hembras_saturadas += 1

        # Convergencia si al menos una hembra tiene 6+ machos
        return hembras_saturadas > 0

    def ejecutar_iteracion(self):
        """Ejecuta una iteración del modelo"""
        N = self.params.tamaño_grilla

        # Encontrar todos los machos y sus movimientos deseados
        movimientos_machos = []
        
        for i in range(N):
            for j in range(N):
                if self.grilla[i, j] == 1:  # Es un macho
                    hembra_cercana, distancia = self.encontrar_hembra_mas_cercana(i, j)
                    
                    if hembra_cercana and distancia <= self.params.radio_atraccion:
                        x_hembra, y_hembra = hembra_cercana
                        dx, dy = self.calcular_direccion_hacia_hembra(i, j, x_hembra, y_hembra)
                        
                        nueva_x = i + dx
                        nueva_y = j + dy
                        
                        # Verificar que la nueva posición esté dentro de límites y vacía
                        if (0 <= nueva_x < N and 0 <= nueva_y < N and 
                            self.grilla[nueva_x, nueva_y] == 0):
                            movimientos_machos.append(((i, j), (nueva_x, nueva_y)))

        # Ejecutar movimientos (aleatorizar para evitar conflictos)
        random.shuffle(movimientos_machos)
        movimientos_realizados = 0

        nueva_grilla = self.grilla.copy()
        
        for (x_old, y_old), (x_new, y_new) in movimientos_machos:
            # Verificar que la posición destino sigue vacía
            if nueva_grilla[x_new, y_new] == 0:
                nueva_grilla[x_new, y_new] = nueva_grilla[x_old, y_old]
                nueva_grilla[x_old, y_old] = 0
                movimientos_realizados += 1

        self.grilla = nueva_grilla
        return movimientos_realizados > 0

    def calcular_metricas(self):
        """Calcula métricas de agrupación y distancia promedio"""
        N = self.params.tamaño_grilla
        
        # Encontrar todas las hembras
        hembras = [(i, j) for i in range(N) for j in range(N) if self.grilla[i, j] == 2]
        
        if not hembras:
            return 0, 0
        
        # Calcular agrupación (promedio de machos alrededor de cada hembra)
        total_machos_alrededor = 0
        distancias_totales = 0
        total_machos = 0
        
        for x_hembra, y_hembra in hembras:
            machos_alrededor = self.contar_machos_alrededor_hembra(x_hembra, y_hembra)
            total_machos_alrededor += machos_alrededor
        
        # Calcular distancia promedio de machos a hembras
        for i in range(N):
            for j in range(N):
                if self.grilla[i, j] == 1:  # Es un macho
                    total_machos += 1
                    hembra_cercana, distancia = self.encontrar_hembra_mas_cercana(i, j)
                    if hembra_cercana:
                        distancias_totales += distancia
        
        agrupacion_promedio = total_machos_alrededor / len(hembras) if hembras else 0
        distancia_promedio = distancias_totales / total_machos if total_machos > 0 else 0
        
        return agrupacion_promedio, distancia_promedio

    def simular(self, mostrar_progreso=True):
        """Ejecuta la simulación completa"""
        print("🐱 Iniciando simulación del Modelo de Atracción de Gatos...")

        # Inicializar población
        self.inicializar_poblacion()

        # Calcular métricas iniciales
        agrupacion_inicial, distancia_inicial = self.calcular_metricas()
        self.historial_agrupacion.append(agrupacion_inicial)
        self.historial_distancia_promedio.append(distancia_inicial)

        if mostrar_progreso:
            print(f"Estado inicial - Agrupación: {agrupacion_inicial:.2f}, Distancia promedio: {distancia_inicial:.2f}")

        # Ejecutar iteraciones
        for iteracion in range(self.params.max_iteraciones):
            self.iteracion_actual = iteracion + 1

            # Ejecutar una iteración
            hubo_movimientos = self.ejecutar_iteracion()

            # Calcular métricas
            agrupacion, distancia = self.calcular_metricas()
            self.historial_agrupacion.append(agrupacion)
            self.historial_distancia_promedio.append(distancia)

            # Verificar convergencia
            if self.verificar_convergencia():
                print(f"\n✓ ¡Convergencia alcanzada! Una o más hembras tienen {self.params.max_machos_por_hembra}+ machos alrededor")
                self.convergencia_alcanzada = True
                break

            # Mostrar progreso cada 10 iteraciones
            if mostrar_progreso and (iteracion + 1) % 10 == 0:
                print(f"Iteración {iteracion + 1:3d} - Agrupación: {agrupacion:.2f}, Distancia: {distancia:.2f}")

            # Si no hay movimientos, la simulación se ha estabilizado
            if not hubo_movimientos:
                print(f"\n⚠️ Simulación estabilizada sin alcanzar convergencia en iteración {iteracion + 1}")
                break

        # Resumen final
        agrupacion_final, distancia_final = self.calcular_metricas()
        print(f"\n📊 RESULTADOS FINALES:")
        print(f"   • Iteraciones ejecutadas: {self.iteracion_actual}")
        print(f"   • Agrupación final: {agrupacion_final:.2f} machos/hembra")
        print(f"   • Distancia promedio final: {distancia_final:.2f}")
        print(f"   • Cambio en agrupación: {agrupacion_final - agrupacion_inicial:+.2f}")
        print(f"   • Cambio en distancia: {distancia_final - distancia_inicial:+.2f}")
        print(f"   • Convergencia: {'Sí' if self.convergencia_alcanzada else 'No'}")

print("✓ Clase ModeloAtraccionGatos implementada")

# Funciones de visualización adaptadas
def visualizar_grilla_gatos(modelo, titulo="Estado de la Simulación de Gatos"):
    """Visualiza el estado actual de la grilla de gatos"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Definir colores: blanco=vacío, azul=macho, rosa=hembra en celo
    colores = ['white', '#3498db', '#e91e63']
    cmap = plt.matplotlib.colors.ListedColormap(colores)

    # Mostrar grilla
    im = ax.imshow(modelo.grilla, cmap=cmap, vmin=0, vmax=2)

    # Agregar números para mostrar agrupación alrededor de hembras
    N = modelo.params.tamaño_grilla
    for i in range(N):
        for j in range(N):
            if modelo.grilla[i, j] == 2:  # Hembra en celo
                machos_alrededor = modelo.contar_machos_alrededor_hembra(i, j)
                ax.text(j, i, str(machos_alrededor), ha='center', va='center', 
                       color='white', fontweight='bold', fontsize=8)

    # Configuración
    ax.set_title(f"{titulo}\nIteración: {modelo.iteracion_actual}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Coordenada X")
    ax.set_ylabel("Coordenada Y")

    # Leyenda
    from matplotlib.patches import Patch
    elementos_leyenda = [Patch(facecolor='white', edgecolor='black', label='Vacío'),
                        Patch(facecolor='#3498db', label='Macho'),
                        Patch(facecolor='#e91e63', label='Hembra en celo')]
    ax.legend(handles=elementos_leyenda, loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()

def visualizar_evolucion_gatos(modelo):
    """Visualiza la evolución de las métricas de gatos a lo largo del tiempo"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    iteraciones = range(len(modelo.historial_agrupacion))

    # Gráfico de agrupación
    ax1.plot(iteraciones, modelo.historial_agrupacion, 'o-', color='#e91e63', linewidth=2, markersize=4)
    ax1.axhline(y=modelo.params.max_machos_por_hembra, color='red', linestyle='--', 
                label=f'Objetivo: {modelo.params.max_machos_por_hembra} machos')
    ax1.set_title('Evolución de la Agrupación', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Iteración')
    ax1.set_ylabel('Machos promedio por hembra')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Gráfico de distancia promedio
    ax2.plot(iteraciones, modelo.historial_distancia_promedio, 'o-', color='#3498db', linewidth=2, markersize=4)
    ax2.set_title('Evolución de la Distancia Promedio', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Iteración')
    ax2.set_ylabel('Distancia promedio macho-hembra')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def analizar_resultados_gatos(modelo):
    """Analiza y muestra estadísticas detalladas del modelo de gatos"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Estado final de la grilla
    colores = ['white', '#3498db', '#e91e63']
    cmap = plt.matplotlib.colors.ListedColormap(colores)

    im = ax1.imshow(modelo.grilla, cmap=cmap, vmin=0, vmax=2)
    ax1.set_title('Estado Final - Distribución de Gatos', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Coordenada X')
    ax1.set_ylabel('Coordenada Y')
    
    # Agregar números de agrupación
    N = modelo.params.tamaño_grilla
    for i in range(N):
        for j in range(N):
            if modelo.grilla[i, j] == 2:  # Hembra en celo
                machos_alrededor = modelo.contar_machos_alrededor_hembra(i, j)
                ax1.text(j, i, str(machos_alrededor), ha='center', va='center', 
                        color='white', fontweight='bold', fontsize=8)

    # Distribución de agentes
    contador = Counter(modelo.grilla.flatten())
    etiquetas = ['Vacío', 'Machos', 'Hembras en celo']
    valores = [contador[0], contador[1], contador[2]]
    colores_bar = ['lightgray', '#3498db', '#e91e63']

    ax2.bar(etiquetas, valores, color=colores_bar)
    ax2.set_title('Distribución de Agentes', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Número de Celdas')

    # Evolución temporal
    iteraciones = range(len(modelo.historial_agrupacion))
    ax3.plot(iteraciones, modelo.historial_agrupacion, 'o-', color='#e91e63',
             linewidth=2, markersize=3, label='Agrupación')
    ax3.axhline(y=modelo.params.max_machos_por_hembra, color='red', linestyle='--', 
                alpha=0.7, label=f'Objetivo: {modelo.params.max_machos_por_hembra}')
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(iteraciones, modelo.historial_distancia_promedio, 'o-', color='#3498db',
                  linewidth=2, markersize=3, label='Distancia')

    ax3.set_title('Evolución Temporal', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Iteración')
    ax3.set_ylabel('Machos por hembra', color='#e91e63')
    ax3_twin.set_ylabel('Distancia promedio', color='#3498db')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')

    # Estadísticas finales
    ax4.axis('off')
    
    # Contar hembras con diferentes niveles de agrupación
    N = modelo.params.tamaño_grilla
    hembras_saturadas = 0
    hembras_con_machos = 0
    total_hembras = 0
    
    for i in range(N):
        for j in range(N):
            if modelo.grilla[i, j] == 2:  # Hembra en celo
                total_hembras += 1
                machos_alrededor = modelo.contar_machos_alrededor_hembra(i, j)
                if machos_alrededor >= modelo.params.max_machos_por_hembra:
                    hembras_saturadas += 1
                if machos_alrededor > 0:
                    hembras_con_machos += 1

    estadisticas = f"""
    ESTADÍSTICAS FINALES - MODELO DE GATOS

    Parámetros:
    • Objetivo: {modelo.params.max_machos_por_hembra} machos por hembra
    • Radio de atracción: {modelo.params.radio_atraccion}
    • Tamaño: {modelo.params.tamaño_grilla}×{modelo.params.tamaño_grilla}

    Resultados:
    • Iteraciones: {modelo.iteracion_actual}
    • Agrupación final: {modelo.historial_agrupacion[-1]:.2f} machos/hembra
    • Distancia promedio: {modelo.historial_distancia_promedio[-1]:.2f}
    • Convergencia: {'Sí' if modelo.convergencia_alcanzada else 'No'}

    Análisis de hembras:
    • Total de hembras: {total_hembras}
    • Hembras con machos cerca: {hembras_con_machos}
    • Hembras "saturadas" (6+ machos): {hembras_saturadas}
    • % de éxito: {(hembras_saturadas/total_hembras*100) if total_hembras > 0 else 0:.1f}%

    Interpretación:
    • Convergencia = Al menos 1 hembra con 6+ machos
    • Mayor agrupación = Más atracción efectiva
    • Menor distancia = Machos más cerca de hembras
    """

    ax4.text(0.05, 0.95, estadisticas, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.show()

print("✓ Funciones de visualización de gatos implementadas")

# Experimento básico
print("🐱 EXPERIMENTO: Simulación de Atracción de Gatos")
print("=" * 55)

modelo_gatos = ModeloAtraccionGatos(params)
modelo_gatos.simular()

print("\n📊 Visualizando resultados...")

# Ejecutar visualizaciones
visualizar_grilla_gatos(modelo_gatos, "Experimento: Atracción de Gatos - Estado Final")
visualizar_evolucion_gatos(modelo_gatos)
analizar_resultados_gatos(modelo_gatos)