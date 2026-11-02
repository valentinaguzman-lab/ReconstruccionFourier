# Reconstrucción de señales con Series de Fourier

Universidad del Cauca - Facultad de Ingeniería Electrónica y Telecomunicaciones
Telecomunicaciones Digitales - Señales y Sistemas

**Integrantes:**
- Nombre Apellido 1
- Nombre Apellido 2
- Nombre Apellido 3

## Descripción

Aplicación con interfaz gráfica (Tkinter) que reconstruye señales periódicas
(cuadrada, triangular y diente de sierra, con periodo T = 2 s y amplitud 1)
a partir de sus series de Fourier, y analiza el efecto del número de
armónicos N en la calidad de la reconstrucción.

La aplicación muestra:
- Señal original
- Señal reconstruida
- Señal de error |x(t) - x̂(t)|
- Espectro de línea
- Métricas: error medio cuadrático (MSE), potencia de la señal original y
  potencia de la señal reconstruida (también por Parseval)

Valores de N disponibles: 1, 2, 3, 5, 10, 20, 50, 100, 150, 200.

## Requisitos

- Python 3.10 o superior (Tkinter viene incluido en el instalador de python.org)
- Librerías: numpy y matplotlib

## Instalación y ejecución

```
pip install -r requirements.txt
python app.py
```

## Estructura

```
app.py              Código fuente (núcleo matemático + interfaz)
requirements.txt    Dependencias
README.md           Este archivo
```

## Uso

1. Seleccione la señal y el número de armónicos N.
2. Ingrese el intervalo de tiempo (t inicial y t final, máximo 100 s).
3. Presione **CALCULAR** (o cambie la señal / N para actualizar al instante).
4. **GENERAR TABLA** muestra MSE y potencias para todos los valores de N.
5. **ANÁLISIS N** muestra las gráficas de MSE y potencia contra N.

## Notas

- Las métricas se calculan sobre un periodo completo T, por lo que no
  dependen del intervalo visualizado.
- El MSE se calcula como el promedio de (x - x̂)². También se muestra el RMSE
  (su raíz cuadrada).
