
import random
import time
import tracemalloc
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import json


def quicksort(arr: List[int]) -> List[int]:
   
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)


def mergesort(arr: List[int]) -> List[int]:
    """
    Implementación de MergeSort recursivo.
    Complejidad: O(n log n) en todos los casos
    """
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    
    return merge(left, right)


def merge(left: List[int], right: List[int]) -> List[int]:
    """Función auxiliar para fusionar dos arreglos ordenados"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def generar_datos(tamano: int, distribucion: str, semilla: int = None) -> List[int]:
    """
    Genera arreglos de datos según especificaciones.
    
    Args:
        tamano: Número de elementos (1000, 10000, 100000)
        distribucion: Tipo de distribución ('aleatoria', 'ordenada', 'inversa')
        semilla: Semilla para reproducibilidad
    
    Returns:
        Lista de enteros según la distribución especificada
    """
    if semilla is not None:
        random.seed(semilla)
    
    if distribucion == 'aleatoria':
        return [random.randint(0, 1000000) for _ in range(tamano)]
    elif distribucion == 'ordenada':
        return list(range(tamano))
    elif distribucion == 'inversa':
        return list(range(tamano, 0, -1))
    else:
        raise ValueError(f"Distribución no válida: {distribucion}")



def medir_ejecucion(algoritmo, datos: List[int]) -> Tuple[float, float]:
    """
    Mide tiempo de ejecución y consumo de memoria de un algoritmo.
    
    Returns:
        Tupla (tiempo_ms, memoria_mb)
    """
    
    datos_copia = datos.copy()
    
  
    tracemalloc.start()
    inicio = time.perf_counter()
    
  
    _ = algoritmo(datos_copia)
    
    fin = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    tiempo_ms = (fin - inicio) * 1000  
    memoria_mb = peak / (1024 * 1024)  
    
    return tiempo_ms, memoria_mb


def ejecutar_experimento(replicas: int = 30) -> pd.DataFrame:
    """
    Ejecuta el experimento completo con todas las combinaciones.
    
    Args:
        replicas: Número de repeticiones por escenario (default: 30)
    
    Returns:
        DataFrame con todos los resultados
    """
    print("=" * 70)
    print("INICIANDO EXPERIMENTO: QuickSort vs MergeSort")
    print("=" * 70)
    
    tamanos = [1000, 10000, 100000]
    distribuciones = ['aleatoria', 'ordenada', 'inversa']
    algoritmos = {
        'QuickSort': quicksort,
        'MergeSort': mergesort
    }
    
    resultados = []
    total_ejecuciones = len(tamanos) * len(distribuciones) * len(algoritmos) * replicas
    ejecucion_actual = 0
    
    for tamano in tamanos:
        for distribucion in distribuciones:
            print(f"\n{'─' * 70}")
            print(f"Tamaño: {tamano:,} | Distribución: {distribucion.capitalize()}")
            print(f"{'─' * 70}")
            
            for replica in range(replicas):
               
                semilla = tamano + hash(distribucion) + replica
                datos = generar_datos(tamano, distribucion, semilla)
                
                for nombre_algo, funcion_algo in algoritmos.items():
                    ejecucion_actual += 1
                    
                    try:
                        tiempo, memoria = medir_ejecucion(funcion_algo, datos)
                        
                        resultados.append({
                            'Algoritmo': nombre_algo,
                            'Tamaño': tamano,
                            'Distribucion': distribucion,
                            'Replica': replica + 1,
                            'Tiempo_ms': tiempo,
                            'Memoria_MB': memoria
                        })
                        
                        
                        if replica == 0:  # Solo mostrar la primera réplica
                            print(f"  {nombre_algo:10s}: {tiempo:8.2f} ms | {memoria:6.2f} MB")
                    
                    except Exception as e:
                        print(f"  ERROR en {nombre_algo}: {str(e)}")
                        continue
                    
                    
                    progreso = (ejecucion_actual / total_ejecuciones) * 100
                    if ejecucion_actual % 60 == 0:  
                        print(f"\n  Progreso total: {progreso:.1f}% ({ejecucion_actual}/{total_ejecuciones})")
    
    print("\n" + "=" * 70)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 70)
    
    return pd.DataFrame(resultados)


def calcular_estadisticas_descriptivas(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula estadísticas descriptivas por grupo"""
    stats_df = df.groupby(['Algoritmo', 'Tamaño', 'Distribucion']).agg({
        'Tiempo_ms': ['mean', 'std', 'min', 'max'],
        'Memoria_MB': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    stats_df.columns = ['_'.join(col).strip('_') for col in stats_df.columns.values]
    
    return stats_df


def realizar_anova(df: pd.DataFrame, distribucion: str):
    """
    Realiza ANOVA de dos vías para una distribución específica.
    """
    print(f"\n{'=' * 70}")
    print(f"ANOVA - Distribución: {distribucion.upper()}")
    print(f"{'=' * 70}")
    
    # Filtrar datos por distribución
    df_filtrado = df[df['Distribucion'] == distribucion]
    
    # ANOVA para Tiempo
    print("\n1. TIEMPO DE EJECUCIÓN:")
    print("-" * 70)
    
    grupos_tiempo = []
    labels = []
    for (algo, tam), group in df_filtrado.groupby(['Algoritmo', 'Tamaño']):
        grupos_tiempo.append(group['Tiempo_ms'].values)
        labels.append(f"{algo}_{tam}")
    
    f_stat_tiempo, p_value_tiempo = stats.f_oneway(*grupos_tiempo)
    print(f"  F-statistic: {f_stat_tiempo:.2f}")
    print(f"  P-value: {p_value_tiempo:.6f}")
    print(f"  Significativo: {'SÍ' if p_value_tiempo < 0.05 else 'NO'} (α = 0.05)")
    
    # ANOVA para Memoria
    print("\n2. CONSUMO DE MEMORIA:")
    print("-" * 70)
    
    grupos_memoria = []
    for (algo, tam), group in df_filtrado.groupby(['Algoritmo', 'Tamaño']):
        grupos_memoria.append(group['Memoria_MB'].values)
    
    f_stat_memoria, p_value_memoria = stats.f_oneway(*grupos_memoria)
    print(f"  F-statistic: {f_stat_memoria:.2f}")
    print(f"  P-value: {p_value_memoria:.6f}")
    print(f"  Significativo: {'SÍ' if p_value_memoria < 0.05 else 'NO'} (α = 0.05)")



def crear_visualizaciones(df: pd.DataFrame, stats_df: pd.DataFrame):
    """Crea gráficos para el informe"""
    
    # Configurar estilo
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
   
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    distribuciones = ['aleatoria', 'ordenada', 'inversa']
    
    for idx, dist in enumerate(distribuciones):
        data = stats_df[stats_df['Distribucion'] == dist]
        
        x = np.arange(len(data['Tamaño'].unique()))
        width = 0.35
        
        qs_data = data[data['Algoritmo'] == 'QuickSort']
        ms_data = data[data['Algoritmo'] == 'MergeSort']
        
        axes[idx].bar(x - width/2, qs_data['Tiempo_ms_mean'], width, 
                     label='QuickSort', alpha=0.8, color='#FF6B6B')
        axes[idx].bar(x + width/2, ms_data['Tiempo_ms_mean'], width,
                     label='MergeSort', alpha=0.8, color='#4ECDC4')
        
        axes[idx].set_xlabel('Tamaño del Arreglo')
        axes[idx].set_ylabel('Tiempo (ms)')
        axes[idx].set_title(f'Distribución {dist.capitalize()}')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(['1K', '10K', '100K'])
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figura1_tiempo_ejecucion.png', dpi=300, bbox_inches='tight')
    print("\n✓ Guardado: figura1_tiempo_ejecucion.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    
    memoria_stats = stats_df.groupby(['Algoritmo', 'Tamaño'])['Memoria_MB_mean'].mean().reset_index()
    
    for algo in ['QuickSort', 'MergeSort']:
        data = memoria_stats[memoria_stats['Algoritmo'] == algo]
        ax.plot(data['Tamaño'], data['Memoria_MB_mean'], 
               marker='o', linewidth=2, markersize=8, label=algo)
    
    ax.set_xlabel('Tamaño del Arreglo')
    ax.set_ylabel('Memoria (MB)')
    ax.set_title('Consumo de Memoria por Tamaño de Arreglo')
    ax.set_xscale('log')
    ax.set_xticks([1000, 10000, 100000])
    ax.set_xticklabels(['1K', '10K', '100K'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figura2_consumo_memoria.png', dpi=300, bbox_inches='tight')
    print("✓ Guardado: figura2_consumo_memoria.png")
    
    plt.close('all')


def generar_tablas_latex(stats_df: pd.DataFrame):
    """Genera tablas en formato LaTeX para el informe"""
    
    print("\n" + "=" * 70)
    print("TABLAS PARA EL INFORME (Formato LaTeX)")
    print("=" * 70)
  
    print("\n--- TABLA 4: Tiempo de Ejecución ---\n")
    
    tabla4 = stats_df[['Tamaño', 'Distribucion', 'Algoritmo', 'Tiempo_ms_mean', 'Tiempo_ms_std']]
    tabla4_pivot = tabla4.pivot_table(
        index=['Tamaño', 'Distribucion'],
        columns='Algoritmo',
        values=['Tiempo_ms_mean', 'Tiempo_ms_std']
    )
    
    for (tam, dist), row in tabla4_pivot.iterrows():
        qs_mean = row['Tiempo_ms_mean']['QuickSort']
        qs_std = row['Tiempo_ms_std']['QuickSort']
        ms_mean = row['Tiempo_ms_mean']['MergeSort']
        ms_std = row['Tiempo_ms_std']['MergeSort']
        
        print(f"{tam:,} & {dist.capitalize()} & {qs_mean:.2f} ± {qs_std:.2f} & {ms_mean:.2f} ± {ms_std:.2f} \\\\")
    
    # Tabla 5: Memoria
    print("\n--- TABLA 5: Consumo de Memoria ---\n")
    
    tabla5 = stats_df.groupby(['Tamaño', 'Algoritmo']).agg({
        'Memoria_MB_mean': 'mean',
        'Memoria_MB_std': 'mean'
    }).reset_index()
    
    tabla5_pivot = tabla5.pivot_table(
        index='Tamaño',
        columns='Algoritmo',
        values=['Memoria_MB_mean', 'Memoria_MB_std']
    )
    
    for tam, row in tabla5_pivot.iterrows():
        qs_mean = row['Memoria_MB_mean']['QuickSort']
        qs_std = row['Memoria_MB_std']['QuickSort']
        ms_mean = row['Memoria_MB_mean']['MergeSort']
        ms_std = row['Memoria_MB_std']['MergeSort']
        
        print(f"{tam:,} & {qs_mean:.2f} ± {qs_std:.2f} & {ms_mean:.2f} ± {ms_std:.2f} \\\\")


def main():
    """Función principal que ejecuta todo el experimento"""
    
    print("\n" + "🔬 " * 20)
    print("EXPERIMENTO COMPARATIVO: QuickSort vs MergeSort")
    print("Universidad Nacional de Trujillo - 2025")
    print("🔬 " * 20 + "\n")
    

    print("\n[FASE 1] Ejecutando experimento...")
    df_resultados = ejecutar_experimento(replicas=30)
 
    df_resultados.to_csv('resultados_completos.csv', index=False)
    print("\n✓ Guardado: resultados_completos.csv")
    
    print("\n[FASE 2] Calculando estadísticas descriptivas...")
    stats_df = calcular_estadisticas_descriptivas(df_resultados)
    stats_df.to_csv('estadisticas_descriptivas.csv', index=False)
    print("✓ Guardado: estadisticas_descriptivas.csv")
    
    print("\n[FASE 3] Realizando análisis ANOVA...")
    for dist in ['aleatoria', 'ordenada', 'inversa']:
        realizar_anova(df_resultados, dist)
   
    print("\n[FASE 4] Generando visualizaciones...")
    crear_visualizaciones(df_resultados, stats_df)
  
    print("\n[FASE 5] Generando tablas para el informe...")
    generar_tablas_latex(stats_df)
    
    print("\n" + "✅ " * 20)
    print("EXPERIMENTO COMPLETADO EXITOSAMENTE")
    print("✅ " * 20)
    
    print("\n📁 ARCHIVOS GENERADOS:")
    print("  • resultados_completos.csv")
    print("  • estadisticas_descriptivas.csv")
    print("  • figura1_tiempo_ejecucion.png")
    print("  • figura2_consumo_memoria.png")
    
    print("\n📊 SIGUIENTE PASO:")
    print("  Usa los datos de 'estadisticas_descriptivas.csv' para")
    print("  reemplazar los valores en las Tablas 4 y 5 de tu informe.")
    
    return df_resultados, stats_df


if __name__ == "__main__":
    
    print("Validando implementaciones...")
    test_data = [64, 34, 25, 12, 22, 11, 90]
    assert quicksort(test_data.copy()) == sorted(test_data), "QuickSort falla"
    assert mergesort(test_data.copy()) == sorted(test_data), "MergeSort falla"
    print("✓ Algoritmos validados correctamente\n")
    
    resultados, estadisticas = main()