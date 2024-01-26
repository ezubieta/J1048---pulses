import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from uncertainties import ufloat
import math
import matplotlib

nice_fonts = {
    # Use LaTeX to write all text
    #"text.usetex": True,
    #"font.family": 'serif',
    "axes.labelsize": 18,
    "font.size": 20,
    "axes.linewidth": 1.5,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "xtick.major.size": 5,  # major tick size in points
    "xtick.minor.size": 4,  # minor tick size in points
    "ytick.major.size": 5.5,  # major tick size in points
    "ytick.minor.size": 3.5,  # minor tick size in points
    "xtick.major.width": 1.4,  # major tick width in points
    "xtick.minor.width": 1.3,  # minor tick width in points
    "ytick.major.width": 1.4,  # major tick width in points
    "ytick.minor.width": 1.3,  # minor tick width in points
}

matplotlib.rcParams.update(nice_fonts)

def chi_cuadrado(y_obs, y_pred, err):
	return np.sum(((y_obs - y_pred) / err)**2)


# Funciones relacionadas con la fase y el ajuste gaussiano
def gaussian_with_constant(x, a, mu, sigma, c,a2,sigma2):
	return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c + a2 * np.exp(-(x - (mu+9.66))**2 / (2 * sigma2**2))

def ajustar(x, y, centro):
	a0 = max(y) - min(y)
	a_ = 0.5 * 10**(np.floor(np.log10(a0)))
	x0 = 180
	s0 = 5
	c0 = min(y)
	c_ = 0.5 * 10**(np.floor(np.log10(c0)))
	initial_guess = [a0, x0, s0, c0,a0,s0]
	boundmin = [0, 170, 0.1, c0 - c_, 0, 0.1]
	boundmax = [1.1*a0, 182, 10, c0 + c_, 1.1*a0, 10]
	params, cov = curve_fit(gaussian_with_constant, x, y, p0=initial_guess, bounds=[boundmin, boundmax])
	errores = np.sqrt(np.diag(cov))
	return params, errores

def date_to_mjd(yyyymmdd):
	year = int(yyyymmdd[:4])
	month = int(yyyymmdd[4:6])
	day = int(yyyymmdd[6:])

	if month == 1 or month == 2:
		yearp = year - 1
		monthp = month + 12
	else:
		yearp = year
		monthp = month
	if ((year < 1582) or (year == 1582 and month < 10) or (year == 1582 and month == 10 and day < 15)):
        # before start of Gregorian calendar
		B = 0
	else:
        # after start of Gregorian calendar
		A = math.trunc(yearp / 100.)
		B = 2 - A + math.trunc(A / 4.)
	if yearp < 0:
		C = math.trunc((365.25 * yearp) - 0.75)
	else:
		C = math.trunc(365.25 * yearp)
		D = math.trunc(30.6001 * (monthp + 1))
		jd = B + C + D + day + 1720994.5

	return jd - 2400000.5

def obtener_fecha(archivo):
	partes = archivo.split("_")
	return partes[3]

# Obtención y ordenación de archivos
carpeta = "./"
archivos_en_carpeta = [archivo for archivo in os.listdir(carpeta) if archivo.startswith("prepfold_timing_")]
archivos_ordenados = sorted(archivos_en_carpeta, key=obtener_fecha)

# Listas para almacenar resultados
fechas, amplitudes, anchos, c, amplitudes2, anchos2 = [], [], [], [], [], []
err_amplitudes, err_anchos, err_c, err_amplitudes2, err_anchos2, chi2 = [], [], [], [], [], []
centro1, centro2=[],[]
err_centro1, err_centro2=[],[]


# Función para generar el ajuste y el gráfico
def generar_ajuste(nombre_archivo):
	bins = []
	signal = []
	ruta_completa = os.path.join(carpeta, nombre_archivo)
	with open(ruta_completa, 'r') as archivo:
		for linea in archivo:
			if not linea.startswith("#") and not linea.isspace():
				columnas = linea.split()
				bins.append(float(columnas[-2]))
				signal.append(float(columnas[-1]))
			if "Data Folded" in linea:
				data_folded = float(linea.split('=')[1].strip())  # Extrae el valor de Data Folded
			elif "T_sample" in linea:
				t_sample = float(linea.split('=')[1].strip())  # Extrae el valor de T_sample


        #signal_v2=savgol_filter(signal, window_length=5, polyorder=3)
	indice_maximo = np.argmax(signal)
	bins = np.roll(bins, -128+indice_maximo)
	fase = bins * 360 / 256
	centro = 180
	indices_filtrados = [i for i, valor in enumerate(fase) if 120 <= valor <= 240]
	fase_v2 = [fase[i] for i in indices_filtrados]
	t_obs=data_folded*t_sample
	signal_v2 = [(signal[i]) for i in indices_filtrados]
	params, errores = ajustar(fase_v2, signal_v2, centro)
	y_pred = gaussian_with_constant(fase_v2, *params)
	ruido = np.std(signal[0:50])
	chi2 = chi_cuadrado(signal_v2, y_pred, ruido)
	dof=len(signal_v2)-6 #number of params
	chisq_red=chi2/dof
	return params[0],params[2], params[3],params[4],params[5],errores[0],errores[2], errores[3], errores[4],errores[5],chisq_red,t_obs, params[1], errores[1]

def generar_grafico(nombre_archivo):
	bins = []
	signal = []
	ruta_completa = os.path.join(carpeta, nombre_archivo)
	with open(ruta_completa, 'r') as archivo:
		for linea in archivo:
			if not linea.startswith("#") and not linea.isspace():
				columnas = linea.split()
				bins.append(float(columnas[-2]))
				signal.append(float(columnas[-1]))
			if "Data Folded" in linea:
				data_folded = float(linea.split('=')[1].strip())  # Extrae el valor de Data Folded
			elif "T_sample" in linea:
				t_sample = float(linea.split('=')[1].strip())  # Extrae el valor de T_sample


	#signal=savgol_filter(signal, window_length=3, polyorder=1)
	indice_maximo = np.argmax(signal)
	bins = np.roll(bins, -128+indice_maximo)
	fase = bins * 360 / 256
	centro = 180
	t_obs=data_folded*t_sample
	indices_filtrados = [i for i, valor in enumerate(fase) if 120 <= valor <= 240]
	fase_v2 = [fase[i] for i in indices_filtrados]
	signal_v2 = [(signal[i]) for i in indices_filtrados]
	params, errores = ajustar(fase_v2, signal_v2, centro)


    # Graficar pulso original
	plt.plot(fase_v2, signal_v2, color='black', label='Pulso original')

    # Graficar la curva ajustada
	x_ajuste = np.linspace(min(fase_v2), max(fase_v2), 1000)
	y_ajuste = gaussian_with_constant(x_ajuste, *params)
	y_gauss1 = gaussian_with_constant(x_ajuste, params[0], params[1],params[2], params[3],0,1)
	y_gauss2 = gaussian_with_constant(x_ajuste, 0,params[1],1, params[3], params[4],params[5])
	plt.plot(x_ajuste, y_ajuste, label='Ajuste gaussiano', color="red")
	plt.plot(x_ajuste,y_gauss1, label='Gaussiana 1', color='blue', linestyle='--')
	plt.plot(x_ajuste, y_gauss2, label='Gaussiana 2', color='green', linestyle='--')
	plt.xlabel('Fase (grados)')
	plt.ylabel('Signal')
	plt.xlim([120,240])
	plt.legend()
	plt.title('Grafico de fase vs. Signal')
	plt.grid(True)
	ruta_png= os.path.join("./pngs", str(obtener_fecha(nombre_archivo)))
	plt.savefig(ruta_png)

# Procesar cada archivo
for archivo in archivos_ordenados:
	fechas.append(date_to_mjd(obtener_fecha(archivo)))
	results = generar_ajuste(archivo)
	amplitudes.append(results[0]/np.sqrt(results[11]))
	anchos.append(results[1])
	c.append(results[2])
	amplitudes2.append(results[3]/np.sqrt(results[11]))
	anchos2.append(results[4])
	err_amplitudes.append(results[5]/np.sqrt(results[11]))
	err_anchos.append(results[6])
	err_c.append(results[7])
	err_amplitudes2.append(results[8]/np.sqrt(results[11]))
	err_anchos2.append(results[9])
	chi2.append(results[10])
	centro1.append(results[12])
	#centro2.append(results[13])
	err_centro1.append(results[13])
	#err_centro2.append(results[15])

# Calcular ac
ac, err_ac = [], []
ac2, err_ac2 = [], []

ratioamp, ratioanc = [], []
err_ratioamp, err_ratioanc=[],[]
dist_centros, err_dist_centros=[],[]

for i in range(len(c)):
	ai = ufloat(amplitudes[i], err_amplitudes[i])
	ci = ufloat(c[i], err_c[i])
	ratio = ai / ci
	ai2 = ufloat(amplitudes2[i], err_amplitudes2[i])
	ratio2 = ai2 / ci
	#if ratio.nominal_value>ratio2.nominal_value:
	#if centro1[i] > centro2[i]:
		#ratio, ratio2 = ratio2, ratio
		#anchos[i],err_anchos[i],anchos2[i], err_anchos2[i], centro1[i], err_centro1[i],centro2[i],err_centro2[i]=anchos2[i],err_anchos2[i],anchos[i], err_anchos[i],centro2[i], err_centro2[i],centro1[i],err_centro1[i]
	ratio_amplitudes = ratio/ratio2
	ratioamp.append(ratio_amplitudes.nominal_value)
	err_ratioamp.append(ratio_amplitudes.s)
	ac.append(ratio.nominal_value)
	err_ac.append(ratio.s)
	ac2.append(ratio2.nominal_value)
	err_ac2.append(ratio2.s)
	ancho1=ufloat(anchos[i],err_anchos[i])
	ancho2=ufloat(anchos2[i], err_anchos2[i])
	ratio_anchos=ancho1/ancho2
	ratioanc.append(ratio_anchos.nominal_value)
	err_ratioanc.append(ratio_anchos.s)
	centroa=ufloat(centro1[i],err_centro1[i])
	centrob=ufloat(centro1[i]+9.66,err_centro1[i])
	distancia=centroa-centrob
	dist_centros.append(np.abs(distancia.nominal_value))
	err_dist_centros.append(distancia.s)

# Gráfico principal
fig, (ax1,ax3,ax5) = plt.subplots(3, 1, figsize=(40,30), sharex=True)
ax1.errorbar(fechas, ac, yerr=err_ac, fmt='o', linestyle='',markersize=4, label="amplitud1/ruido", color="blue")
ax1.grid()
ax1.set_ylabel("amplitud (a.u.)")
ax1.set_title("Gaussiana 1")
ax1.axvspan(59200, 59206, color='gray', alpha=0.5)
ax1.axvspan(59530, 59550, color='gray', alpha=0.5)
ax1.axvspan(60060,60080 , color='gray', alpha=0.5)
ax1.axvspan(60245, 60265, color='gray', alpha=0.5)
ax2=ax1.twinx()
ax2.errorbar(fechas, anchos, yerr=err_anchos, fmt='o', linestyle=' ' , markersize=4,label="anchos1", color='orange')
ax2.set_ylabel("ancho (º)")
ax1.legend(loc='upper right')
ax2.legend(loc='lower left')
ax3.errorbar(fechas, ac2, yerr=err_ac2,fmt='o', linestyle='', markersize=4, label="amplitud2/ruido", color="blue")
ax3.grid()
ax3.set_title("Gaussiana 2")
ax3.set_ylabel("amplitud (a.u.)")
ax3.axvspan(59200, 59206, color='gray', alpha=0.5)
ax3.axvspan(59530, 59550, color='gray', alpha=0.5)
ax3.axvspan(60060,60080 , color='gray', alpha=0.5)
ax3.axvspan(60245, 60265, color='gray', alpha=0.5)
ax4=ax3.twinx()
ax4.errorbar(fechas, anchos2, yerr=err_anchos2,fmt='o', linestyle=' ',markersize=4, label="anchos2", color='orange')
ax3.legend(loc='upper right')
ax4.legend(loc='lower left')
ax4.set_ylabel("ancho (º)")
ax5.errorbar(fechas,ratioamp,yerr=err_ratioamp, fmt='o', linestyle=' ', markersize=4 , label='ratio amplitudes', color='black')
ax5.grid()
ax5.tick_params(axis='x', labelsize=12)
ax5.tick_params(axis='y', labelsize=12)
ax5.set_ylabel("Ratio")
ax5.set_title("Comparacion")
ax5.errorbar(fechas,ratioanc,yerr=err_ratioanc, fmt='o', linestyle=' ', markersize=4 , label='ratio anchos', color='green')
ax5.axvspan(59200, 59206, color='gray', alpha=0.5)
ax5.axvspan(59530, 59550, color='gray', alpha=0.5)
ax5.axvspan(60060,60080 , color='gray', alpha=0.5)
ax5.axvspan(60245, 60265, color='gray', alpha=0.5)
ax5.legend(loc='upper right')
ax5.set_ylim([0,5])


# Guardar gráfico
plt.savefig("gaussianas.pdf")
plt.show()
plt.close()


plt.figure(figsize=(10, 8))
scatter1 = plt.scatter(ac, anchos, c=fechas, cmap='viridis', marker='o',alpha=0.8, label="Gaussiana 1")
plt.colorbar(scatter1, label='Fechas')
scatter2 = plt.scatter(ac2, anchos2, c=fechas, cmap='viridis', marker='x',alpha=0.8, label="Gaussiana 2")
plt.xlabel('Amplitud')
plt.ylabel('Ancho')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("cmap.pdf")
plt.show()
plt.close()



plt.plot(fechas,chi2, label="chi_sq")
plt.axvspan(59200, 59206, color='gray', alpha=0.5)
plt.axvspan(59530, 59550, color='gray', alpha=0.5)
plt.axvspan(60060,60080 , color='gray', alpha=0.5)
plt.axvspan(60245, 60265, color='gray', alpha=0.5)
plt.grid()
plt.legend()
plt.show()
plt.savefig("chisq.pdf")
plt.close()

plt.errorbar(fechas, dist_centros, yerr=err_dist_centros, markersize=4, linestyle=' ', fmt='o',label="Distancia entre centros", color='red')
plt.axvspan(59200, 59206, color='gray', alpha=0.5)
plt.axvspan(59530, 59550, color='gray', alpha=0.5)
plt.axvspan(60060,60080 , color='gray', alpha=0.5)
plt.axvspan(60245, 60265, color='gray', alpha=0.5)
plt.grid()
plt.ylabel("Distancia (º)")
plt.legend()
plt.show()
plt.savefig("distancia_centros")
plt.close()

# Generar GIF animado
def actualizar(frame):
	plt.clf()
	generar_grafico(archivos_ordenados[frame])
	fecha = date_to_mjd(obtener_fecha(archivos_ordenados[frame]))
	tipo = archivos_ordenados[frame].split("_")[3]
	plt.annotate(u"Fecha: {}".format(fecha), (0.02, 0.95), xycoords='axes fraction', fontsize=10)

animacion = FuncAnimation(plt.gcf(), actualizar, frames=len(archivos_ordenados), repeat=False)
ruta_gif = "2gaussianas.gif"
animacion.save(ruta_gif, writer="pillow", fps=0.666, dpi=80, savefig_kwargs={'facecolor': 'white'})

import pandas as pd
df = pd.DataFrame({
    'fechas': fechas,
    'SN1': ac,
    'err_SN1': err_ac,
    'ancho1': anchos,
    'err_ancho1': err_anchos,
    'SN2': ac2,
    'err_SN2': err_ac2,
    'ancho2': anchos2,
    'err_ancho2': err_anchos2,
    'ratioamp': ratioamp,
    'err_ratioamp': err_ratioamp,
    'ratioanc': ratioanc,
    'err_ratioanc': err_ratioanc,
    'dist_centros': dist_centros,
    'err_dist_centros': err_dist_centros
})

# Guardar el DataFrame en un archivo csv
df.to_csv('resultados.csv', index=False)
