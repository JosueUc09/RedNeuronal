
import csv
import json
import math
import random
from pathlib import Path

DATOS = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
POBLACION, GENERACIONES = 80, 400
N_GENES, ELITE = 13, 2
P_CRUZA, P_MUTACION, SIGMA = 0.9, 0.15, 0.7
LIMITE = 10.0


def sigmoide(z):
    return 1.0 / (1.0 + math.exp(-z))


def predecir(c, x1, x2):
    # Tres neuronas ocultas: dos pesos y un sesgo cada una.
    h = [sigmoide(c[3*j]*x1 + c[3*j+1]*x2 + c[3*j+2])
         for j in range(3)]
    # Tres pesos de salida y un sesgo final.
    return sigmoide(sum(c[9+j]*h[j] for j in range(3)) + c[12])


def error(c):
    return sum((predecir(c, x1, x2) - y)**2
               for x1, x2, y in DATOS) / len(DATOS)


def aptitud(c):
    return 1.0 / (1.0 + error(c))


def exactitud(c):
    aciertos = sum(int(predecir(c, a, b) >= 0.5) == y
                   for a, b, y in DATOS)
    return aciertos / len(DATOS)


def ejecutar(semilla):
    rng = random.Random(semilla)
    poblacion = [[rng.uniform(-2, 2) for _ in range(N_GENES)]
                 for _ in range(POBLACION)]
    historial, inicial = [], None
    for generacion in range(GENERACIONES + 1):
        evaluada = sorted((error(c), c) for c in poblacion)
        historial.append(evaluada[0][0])
        if generacion == 0:
            inicial = evaluada[0][1][:]
        if generacion == GENERACIONES:
            break
        nueva = [c[:] for _, c in evaluada[:ELITE]]
        while len(nueva) < POBLACION:
            # Menor error equivale a mayor aptitud.
            p1 = min(rng.sample(evaluada, 3), key=lambda t: t[0])[1]
            p2 = min(rng.sample(evaluada, 3), key=lambda t: t[0])[1]
            if rng.random() < P_CRUZA:
                hijo = [a if rng.random() < 0.5 else b
                        for a, b in zip(p1, p2)]
            else:
                hijo = p1[:]
            for j in range(N_GENES):
                if rng.random() < P_MUTACION:
                    hijo[j] += rng.gauss(0, SIGMA)
                hijo[j] = max(-LIMITE, min(LIMITE, hijo[j]))
            assert len(hijo) == 13
            assert all(-LIMITE <= g <= LIMITE for g in hijo)
            nueva.append(hijo)
        poblacion = nueva
    assert all(b <= a + 1e-12
               for a, b in zip(historial, historial[1:]))
    return inicial, evaluada[0][1], historial


def main():
    carpeta = Path(__file__).resolve().parent
    filas, detalles = [], {}
    for semilla in range(10):
        inicial, final, historial = ejecutar(semilla)
        filas.append({
            'semilla': semilla, 'ecm_inicial': error(inicial),
            'ecm_final': error(final),
            'exactitud_inicial': exactitud(inicial),
            'exactitud_final': exactitud(final)})
        detalles[str(semilla)] = {
            'pesos_iniciales': inicial, 'pesos_finales': final,
            'historial': historial,
            'predicciones': [predecir(final, a, b) for a, b, _ in DATOS]}
    with (carpeta / 'resultados_red.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=filas[0].keys())
        w.writeheader()
        w.writerows(filas)
    (carpeta / 'resultados_red.json').write_text(
        json.dumps({'resumen': filas, 'detalles': detalles}, indent=2),
        encoding='utf-8')
    lineas = ['Semilla | ECM inicial | ECM final | Exactitud final']
    for f in filas:
        lineas.append(f"{f['semilla']:7d} | {f['ecm_inicial']:.8f} | "
                      f"{f['ecm_final']:.8f} | "
                      f"{100*f['exactitud_final']:.0f}%")
    lineas.append('\nPredicciones finales, semilla 0:')
    for (a, b, y), pred in zip(DATOS, detalles['0']['predicciones']):
        lineas.append(f'{a} XOR {b}: {pred:.6f}; '
                      f'clase={int(pred >= 0.5)}; esperado={y}')
    salida = '\n'.join(lineas)
    print(salida)
    (carpeta / 'salida_ejecucion.txt').write_text(salida, encoding='utf-8')


if __name__ == '__main__':
    main()

    from generar_graficas import main as generar_graficas
    generar_graficas()