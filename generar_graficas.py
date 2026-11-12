
import json
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # Guarda imagenes sin necesitar una ventana.
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit('Instala Matplotlib: python -m pip install matplotlib')


def main():
    carpeta = Path(__file__).resolve().parent
    archivo = carpeta / 'resultados_red.json'
    if not archivo.exists():
        raise SystemExit(
            'Falta resultados_red.json. Ejecuta primero: '
            'python red_neuronal_genetica.py')
    datos = json.loads(archivo.read_text(encoding='utf-8'))
    detalles = datos['detalles']
    salida = carpeta / 'graficas'
    salida.mkdir(exist_ok=True)
    plt.rcParams.update({
        'font.size': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    # Figura 1: mismas semillas y escala que en el informe.
    fig, ax = plt.subplots(figsize=(8, 3.15))
    for semilla in ['0', '1', '3']:
        historial = detalles[semilla]['historial']
        ax.plot(range(len(historial)), historial,
                label='Semilla ' + semilla, linewidth=1.8)
    ax.set_xlabel('Generación (0 = población inicial)')
    ax.set_ylabel('Mejor ECM')
    ax.set_yscale('log')
    ax.set_xlim(0, len(detalles['0']['historial']) - 1)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    ruta1 = salida / 'Figura_1_Evolucion_del_error.png'
    fig.savefig(ruta1, dpi=300, facecolor='white')
    plt.close(fig)

    # Figura 2: salidas continuas de la semilla 1, sin aplicar umbral.
    etiquetas = ['(0, 0)', '(0, 1)', '(1, 0)', '(1, 1)']
    esperadas = [0, 1, 1, 0]
    predicciones = detalles['1']['predicciones']
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.bar([x - 0.18 for x in range(4)], esperadas, width=0.36,
           color='#b8cad5', label='Salida esperada')
    ax.bar([x + 0.18 for x in range(4)], predicciones, width=0.36,
           color='#125569', label='Salida de la red')
    ax.set_xticks(range(4))
    ax.set_xticklabels(etiquetas)
    ax.set_ylabel('Salida')
    ax.set_xlabel('Entradas XOR')
    ax.set_ylim(0, 1.2)
    ax.legend(ncol=2)
    fig.tight_layout()
    ruta2 = salida / 'Figura_2_Comparacion_de_salidas.png'
    fig.savefig(ruta2, dpi=300, facecolor='white')
    plt.close(fig)
    print('Graficas generadas correctamente:')
    print(ruta1)
    print(ruta2)


if __name__ == '__main__':
    main()
