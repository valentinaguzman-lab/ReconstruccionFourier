"""
Reconstrucción de señales periódicas mediante Series de Fourier.
Universidad del Cauca - Señales y Sistemas

Señales: cuadrada, triangular y diente de sierra (periodo T, amplitud 1).
La reconstrucción usa los coeficientes analíticos de la serie de Fourier,
sin funciones de alto nivel (scipy.signal, fft, etc.).

Para ejecutar:  python app.py
"""

import math
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ==============================================================
# CONSTANTES
# ==============================================================

# Periodo de todas las señales y valores derivados
PERIODO_T = 2.0                              # s
FRECUENCIA_F0 = 1.0 / PERIODO_T              # Hz (frecuencia fundamental)
FRECUENCIA_W0 = 2.0 * math.pi / PERIODO_T    # rad/s

# Valores de armónicos exigidos por los requerimientos
ARMONICOS_VALORES = [1, 2, 3, 5, 10, 20, 50, 100, 150, 200]

# Nombres de las señales
SENAL_CUADRADA = "Señal Cuadrada"
SENAL_TRIANGULAR = "Señal Triangular"
SENAL_SIERRA = "Señal Diente de Sierra"
SENALES = [SENAL_CUADRADA, SENAL_TRIANGULAR, SENAL_SIERRA]

# Intervalo de visualización por defecto y límites permitidos
T_INICIAL_DEFECTO = "0"
T_FINAL_DEFECTO = "6"
DURACION_MINIMA = 0.01       # s
DURACION_MAXIMA = 100.0      # s
LIMITE_TIEMPO = 1000.0       # s (valor absoluto máximo de ti y tf)

# Muestreo para visualizar (se ajusta para evitar aliasing)
MUESTRAS_POR_CICLO = 40      # muestras por ciclo del armónico más alto
MUESTRAS_MINIMAS = 2000
MUESTRAS_MAXIMAS = 300000

# Muestras (en un periodo T) para calcular las métricas
MUESTRAS_METRICAS = 20000


# ==============================================================
# NÚCLEO MATEMÁTICO (sin interfaz gráfica)
# ==============================================================

def validar_tipo(tipo):
    """Lanza ValueError si el tipo de señal no existe."""
    if tipo not in SENALES:
        raise ValueError(f"Tipo de señal no reconocido: '{tipo}'.")


def generar_senal(t, tipo):
    """Señal original x(t) de periodo T y amplitud 1."""
    validar_tipo(tipo)
    ciclos = t / PERIODO_T

    if tipo == SENAL_CUADRADA:
        fase = ciclos - np.floor(ciclos)          # fase en [0, 1)
        return np.where(fase < 0.5, 1.0, -1.0)

    if tipo == SENAL_TRIANGULAR:
        return 1.0 - 4.0 * np.abs(ciclos - np.floor(ciclos + 0.5))

    # Diente de sierra
    return 2.0 * (ciclos - np.floor(ciclos + 0.5))


def coeficientes_serie(tipo, N):
    """
    Coeficientes de la serie de Fourier hasta el armónico N.

    Devuelve (n, coef, base):
      n    : índices de armónicos 1..N
      coef : coeficiente de cada armónico
      base : función base (np.sin o np.cos) que multiplica a cada coeficiente

    Cuadrada:    b_n = 4/(n*pi)             (n impar, senos)
    Triangular:  a_n = 8/(n*pi)^2           (n impar, cosenos)
    Sierra:      b_n = 2*(-1)^(n+1)/(n*pi)  (todo n, senos)
    """
    validar_tipo(tipo)
    n = np.arange(1, N + 1)
    es_impar = (n % 2 == 1)

    if tipo == SENAL_CUADRADA:
        coef = np.where(es_impar, 4.0 / (np.pi * n), 0.0)
        base = np.sin
    elif tipo == SENAL_TRIANGULAR:
        coef = np.where(es_impar, 8.0 / (np.pi * n) ** 2, 0.0)
        base = np.cos
    else:
        coef = 2.0 * (-1.0) ** (n + 1) / (np.pi * n)
        base = np.sin

    return n, coef, base


def reconstruir_serie(t, tipo, N):
    """Señal reconstruida x_hat(t) con N armónicos."""
    n, coef, base = coeficientes_serie(tipo, N)
    x_rec = np.zeros_like(t, dtype=float)

    for ni, ci in zip(n, coef):
        if ci != 0.0:
            x_rec += ci * base(ni * FRECUENCIA_W0 * t)

    return x_rec


def calcular_mse(x, x_rec):
    """Error medio cuadrático: promedio de (x - x_hat)^2."""
    return float(np.mean((x - x_rec) ** 2))


def calcular_potencia(x):
    """Potencia promedio de una señal: promedio de x^2."""
    return float(np.mean(x ** 2))


def potencia_parseval(coef):
    """Potencia desde los coeficientes (Parseval): suma de c_n^2 / 2."""
    return float(np.sum(coef ** 2) / 2.0)


def tiempo_metricas():
    """Un periodo T, con puntos en el centro de cada intervalo."""
    k = np.arange(MUESTRAS_METRICAS)
    return (k + 0.5) * PERIODO_T / MUESTRAS_METRICAS


def calcular_metricas(tipo, N):
    """
    Métricas sobre un periodo completo (no dependen del intervalo
    visualizado, así los resultados son siempre los mismos).
    """
    t = tiempo_metricas()
    x = generar_senal(t, tipo)
    x_rec = reconstruir_serie(t, tipo, N)
    _, coef, _ = coeficientes_serie(tipo, N)

    mse = calcular_mse(x, x_rec)
    return {
        "N": N,
        "MSE": mse,
        "RMSE": math.sqrt(mse),
        "P_original": calcular_potencia(x),
        "P_reconstruida": calcular_potencia(x_rec),
        "P_parseval": potencia_parseval(coef),
    }


def numero_muestras(duracion, N):
    """Muestras necesarias para ver bien el armónico N (evita aliasing)."""
    n = int(MUESTRAS_POR_CICLO * N * FRECUENCIA_F0 * duracion)
    return max(MUESTRAS_MINIMAS, min(n, MUESTRAS_MAXIMAS))


# ==============================================================
# VALIDACIÓN DE ENTRADAS
# ==============================================================

def validar_intervalo(texto_ti, texto_tf):
    """Convierte y valida el intervalo de tiempo. Devuelve (ti, tf)."""
    try:
        ti = float(texto_ti.strip().replace(",", "."))
        tf = float(texto_tf.strip().replace(",", "."))
    except ValueError:
        raise ValueError(
            "Los tiempos inicial y final deben ser números "
            "(por ejemplo 0 y 6)."
        )

    if not (math.isfinite(ti) and math.isfinite(tf)):
        raise ValueError("Los tiempos deben ser números finitos.")

    if abs(ti) > LIMITE_TIEMPO or abs(tf) > LIMITE_TIEMPO:
        raise ValueError(
            f"Los tiempos deben estar entre -{LIMITE_TIEMPO:.0f} s "
            f"y {LIMITE_TIEMPO:.0f} s."
        )

    if tf <= ti:
        raise ValueError(
            "El tiempo final debe ser mayor que el tiempo inicial."
        )

    duracion = tf - ti
    if duracion < DURACION_MINIMA:
        raise ValueError(
            f"El intervalo es muy pequeño (mínimo {DURACION_MINIMA} s)."
        )
    if duracion > DURACION_MAXIMA:
        raise ValueError(
            f"El intervalo es demasiado grande (máximo "
            f"{DURACION_MAXIMA:.0f} s)."
        )

    return ti, tf


def validar_armonicos(texto):
    """Convierte y valida el número de armónicos N."""
    try:
        N = int(texto)
    except ValueError:
        raise ValueError("Seleccione un número de armónicos N de la lista.")

    if N not in ARMONICOS_VALORES:
        raise ValueError(
            "N debe ser uno de: "
            + ", ".join(str(v) for v in ARMONICOS_VALORES)
        )
    return N


# ==============================================================
# INTERFAZ GRÁFICA
# ==============================================================

class FourierApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Reconstrucción de Señales - Series de Fourier")
        self.root.geometry("1250x850")
        self.root.minsize(1100, 750)
        self.root.protocol("WM_DELETE_WINDOW", self.cerrar)

        self.crear_interfaz()

        # Valores iniciales
        self.cb_senal.current(0)
        self.cb_armonicos.current(ARMONICOS_VALORES.index(10))  # N = 10

        self.actualizar_graficas()

    def cerrar(self):
        """Cierra la aplicación por completo."""
        self.root.quit()
        self.root.destroy()

    # ----------------------------------------------------------
    # Construcción de la interfaz
    # ----------------------------------------------------------

    def crear_interfaz(self):

        tk.Label(
            self.root,
            text="RECONSTRUCCIÓN DE SEÑALES MEDIANTE SERIES DE FOURIER",
            font=("Arial", 16, "bold")
        ).pack(pady=8)

        tk.Label(
            self.root,
            text=f"Periodo T = {PERIODO_T:.1f} s    |    "
                 f"Frecuencia fundamental f₀ = {FRECUENCIA_F0:.2f} Hz",
            font=("Arial", 10)
        ).pack(pady=2)

        # ---------------- Controles ----------------
        control_frame = tk.LabelFrame(
            self.root, text="Parámetros de simulación", padx=10, pady=8
        )
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        tk.Label(control_frame, text="Señal:").grid(row=0, column=0, padx=5)
        self.cb_senal = ttk.Combobox(
            control_frame, values=SENALES, state="readonly", width=24
        )
        self.cb_senal.grid(row=0, column=1, padx=5)

        tk.Label(control_frame, text="N Armónicos:").grid(
            row=0, column=2, padx=5
        )
        self.cb_armonicos = ttk.Combobox(
            control_frame, values=ARMONICOS_VALORES, state="readonly",
            width=10
        )
        self.cb_armonicos.grid(row=0, column=3, padx=5)

        tk.Label(control_frame, text="t inicial (s):").grid(
            row=0, column=4, padx=5
        )
        self.entry_ti = ttk.Entry(control_frame, width=8)
        self.entry_ti.insert(0, T_INICIAL_DEFECTO)
        self.entry_ti.grid(row=0, column=5, padx=5)

        tk.Label(control_frame, text="t final (s):").grid(
            row=0, column=6, padx=5
        )
        self.entry_tf = ttk.Entry(control_frame, width=8)
        self.entry_tf.insert(0, T_FINAL_DEFECTO)
        self.entry_tf.grid(row=0, column=7, padx=5)

        # Actualización inmediata al cambiar señal, N o presionar Enter
        self.cb_senal.bind("<<ComboboxSelected>>", self.actualizar_graficas)
        self.cb_armonicos.bind(
            "<<ComboboxSelected>>", self.actualizar_graficas
        )
        self.entry_ti.bind("<Return>", self.actualizar_graficas)
        self.entry_tf.bind("<Return>", self.actualizar_graficas)

        # ---------------- Botones ----------------
        botones = tk.Frame(control_frame)
        botones.grid(row=1, column=0, columnspan=8, pady=(8, 0))

        tk.Button(
            botones, text="CALCULAR", command=self.actualizar_graficas,
            bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), padx=12
        ).pack(side=tk.LEFT, padx=8)

        tk.Button(
            botones, text="GENERAR TABLA", command=self.generar_tabla,
            font=("Arial", 10, "bold"), padx=12
        ).pack(side=tk.LEFT, padx=8)

        tk.Button(
            botones, text="ANÁLISIS N", command=self.mostrar_analisis,
            font=("Arial", 10, "bold"), padx=12
        ).pack(side=tk.LEFT, padx=8)

        # ---------------- Métricas ----------------
        metricas_frame = tk.LabelFrame(
            self.root,
            text="Métricas de desempeño (calculadas sobre un periodo T)",
            padx=10, pady=5
        )
        metricas_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        fuente = ("Arial", 10, "bold")
        self.lbl_mse = tk.Label(metricas_frame, text="MSE: ---", font=fuente)
        self.lbl_rmse = tk.Label(metricas_frame, text="RMSE: ---", font=fuente)
        self.lbl_pot_orig = tk.Label(
            metricas_frame, text="P. original: ---", font=fuente
        )
        self.lbl_pot_rec = tk.Label(
            metricas_frame, text="P. reconstruida: ---", font=fuente
        )
        self.lbl_pot_par = tk.Label(
            metricas_frame, text="P. Parseval: ---", font=fuente
        )

        for etiqueta in (self.lbl_mse, self.lbl_rmse, self.lbl_pot_orig,
                         self.lbl_pot_rec, self.lbl_pot_par):
            etiqueta.pack(side=tk.LEFT, padx=12)

        # ---------------- Figura con 4 gráficas ----------------
        self.fig = Figure(figsize=(11, 6.5), dpi=100)
        self.ax1 = self.fig.add_subplot(221)
        self.ax2 = self.fig.add_subplot(222)
        self.ax3 = self.fig.add_subplot(223)
        self.ax4 = self.fig.add_subplot(224)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(
            side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=5
        )

    # ----------------------------------------------------------
    # Lectura de parámetros
    # ----------------------------------------------------------

    def obtener_parametros(self):
        """Lee y valida lo que ingresó el usuario."""
        tipo = self.cb_senal.get()
        validar_tipo(tipo)
        N = validar_armonicos(self.cb_armonicos.get())
        ti, tf = validar_intervalo(self.entry_ti.get(), self.entry_tf.get())
        return tipo, N, ti, tf

    # ----------------------------------------------------------
    # Actualizar las 4 gráficas y las métricas
    # ----------------------------------------------------------

    def actualizar_graficas(self, evento=None):

        try:
            tipo, N, ti, tf = self.obtener_parametros()

            # Señales en el intervalo pedido
            t = np.linspace(ti, tf, numero_muestras(tf - ti, N))
            x_orig = generar_senal(t, tipo)
            x_rec = reconstruir_serie(t, tipo, N)
            error = np.abs(x_orig - x_rec)

            # Espectro de línea y métricas
            n, coef, _ = coeficientes_serie(tipo, N)
            frecuencias = n * FRECUENCIA_F0
            amplitudes = np.abs(coef)
            m = calcular_metricas(tipo, N)

            for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
                ax.clear()

            # 1. Señal original
            self.ax1.plot(t, x_orig, linewidth=1.5)
            self.ax1.set_title("1. Señal Original")
            self.ax1.set_xlabel("Tiempo (s)")
            self.ax1.set_ylabel("Amplitud")
            self.ax1.grid(True)

            # 2. Señal reconstruida
            self.ax2.plot(t, x_rec, linewidth=1.5)
            self.ax2.set_title(f"2. Señal Reconstruida - N = {N}")
            self.ax2.set_xlabel("Tiempo (s)")
            self.ax2.set_ylabel("Amplitud")
            self.ax2.grid(True)

            # 3. Error |x(t) - x_hat(t)|
            self.ax3.plot(t, error, linewidth=1.2)
            self.ax3.set_title("3. Señal de Error |x(t) - x̂(t)|")
            self.ax3.set_xlabel("Tiempo (s)")
            self.ax3.set_ylabel("|e(t)|")
            self.ax3.grid(True)

            # 4. Espectro de línea
            self.ax4.stem(frecuencias, amplitudes, basefmt=" ")
            self.ax4.set_title("4. Espectro de Línea (múltiplos de f₀)")
            self.ax4.set_xlabel("Frecuencia (Hz)")
            self.ax4.set_ylabel("|Coeficiente|")
            self.ax4.grid(True)

            # Métricas
            self.lbl_mse.config(text=f"MSE: {m['MSE']:.4e}")
            self.lbl_rmse.config(text=f"RMSE: {m['RMSE']:.4e}")
            self.lbl_pot_orig.config(
                text=f"P. original: {m['P_original']:.6f}"
            )
            self.lbl_pot_rec.config(
                text=f"P. reconstruida: {m['P_reconstruida']:.6f}"
            )
            self.lbl_pot_par.config(
                text=f"P. Parseval: {m['P_parseval']:.6f}"
            )

            self.fig.tight_layout()
            self.canvas.draw()

        except ValueError as e:
            messagebox.showerror("Error de entrada", str(e))
        except Exception as e:
            messagebox.showerror(
                "Error", f"Ocurrió un error durante el cálculo:\n\n{e}"
            )

    # ----------------------------------------------------------
    # Resultados para todos los valores de N
    # ----------------------------------------------------------

    def calcular_resultados(self):
        tipo = self.cb_senal.get()
        validar_tipo(tipo)
        return [calcular_metricas(tipo, N) for N in ARMONICOS_VALORES]

    # ----------------------------------------------------------
    # Ventana con la tabla de resultados
    # ----------------------------------------------------------

    def generar_tabla(self):

        try:
            resultados = self.calcular_resultados()

            ventana = tk.Toplevel(self.root)
            ventana.title("Tabla de Resultados - Series de Fourier")
            ventana.geometry("1000x500")

            tk.Label(
                ventana,
                text=f"TABLA DE RESULTADOS\n{self.cb_senal.get()}",
                font=("Arial", 14, "bold")
            ).pack(pady=10)

            columnas = ("N", "MSE", "RMSE", "P_original",
                        "P_reconstruida", "P_parseval")
            encabezados = ("N Armónicos", "MSE", "RMSE",
                           "Potencia Original", "Potencia Reconstruida",
                           "Potencia (Parseval)")
            anchos = (100, 150, 150, 160, 180, 170)

            tabla = ttk.Treeview(
                ventana, columns=columnas, show="headings", height=12
            )
            for col, texto, ancho in zip(columnas, encabezados, anchos):
                tabla.heading(col, text=texto)
                tabla.column(col, width=ancho, anchor="center")

            for r in resultados:
                tabla.insert("", "end", values=(
                    r["N"],
                    f"{r['MSE']:.6e}",
                    f"{r['RMSE']:.6e}",
                    f"{r['P_original']:.8f}",
                    f"{r['P_reconstruida']:.8f}",
                    f"{r['P_parseval']:.8f}",
                ))

            tabla.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

            tk.Label(
                ventana,
                text="La tabla muestra cómo cambian el error y la potencia "
                     "al aumentar el número de armónicos.",
                font=("Arial", 10)
            ).pack(pady=8)

        except Exception as e:
            messagebox.showerror(
                "Error", f"No se pudo generar la tabla:\n\n{e}"
            )

    # ----------------------------------------------------------
    # Ventana con gráficas de análisis (MSE y potencia vs N)
    # ----------------------------------------------------------

    def mostrar_analisis(self):

        try:
            resultados = self.calcular_resultados()
            N = np.array([r["N"] for r in resultados])
            mse = np.array([r["MSE"] for r in resultados])
            p_orig = np.array([r["P_original"] for r in resultados])
            p_rec = np.array([r["P_reconstruida"] for r in resultados])

            ventana = tk.Toplevel(self.root)
            ventana.title("Análisis de Resultados")
            ventana.geometry("1100x700")

            fig = Figure(figsize=(10, 6), dpi=100)

            ax1 = fig.add_subplot(211)
            ax1.plot(N, mse, marker="o", linewidth=1.5)
            ax1.set_title(
                f"Error Medio Cuadrático vs N - {self.cb_senal.get()}"
            )
            ax1.set_xlabel("Número de armónicos N")
            ax1.set_ylabel("MSE")
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.grid(True, which="both")

            ax2 = fig.add_subplot(212)
            ax2.plot(N, p_orig, marker="o", linewidth=1.5,
                     label="Potencia original")
            ax2.plot(N, p_rec, marker="s", linewidth=1.5,
                     label="Potencia reconstruida")
            ax2.set_title("Potencia vs N")
            ax2.set_xlabel("Número de armónicos N")
            ax2.set_ylabel("Potencia")
            ax2.set_xscale("log")
            ax2.grid(True, which="both")
            ax2.legend()

            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=ventana)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            canvas.draw()
            ventana.canvas = canvas   # evita que Python lo elimine

        except Exception as e:
            messagebox.showerror(
                "Error", f"No se pudo generar el análisis:\n\n{e}"
            )


# ==============================================================
# PUNTO DE ENTRADA
# ==============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = FourierApp(root)
    root.mainloop()
