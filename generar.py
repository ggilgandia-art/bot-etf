#!/usr/bin/env python3
"""Genera index.html con los datos actualizados del bot de seguimiento de ETF."""

import json
import math
import os
import random
import re
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

TICKERS = {
    "RV_MUNDIAL":   "EUNL.DE",
    "RV_EMERGENTE": "IS3N.DE",
    "BONOS_EUR":    "XGLE.DE",
    "ORO":          "4GLD.DE",
}
ETF_NOMBRES = {
    "RV_MUNDIAL":   "EUNL en Xetra (mismo fondo)",
    "RV_EMERGENTE": "IS3N en Xetra (mismo fondo)",
    "BONOS_EUR":    "XGLE en Xetra (acumulación)",
    "ORO":          "4GLD en Xetra",
}

CAPITAL_INICIAL = 1000.0
COMISION = 2.0        # euros por orden
SMA_PERIODOS = 10     # meses
INICIO_BACKTEST = "2004-01-01"   # necesitamos 10 meses de lookback antes de 2005
INICIO_HISTORIA = "2005-01-01"   # primer mes con señal válida
UMBRAL = -0.0248      # el bot no puede quedar más de 2,48 pp por debajo de la fija ajustada
EXPOSICION_DEV = 0.668  # fracción media del capital invertida (para ajustar cartera fija)

# Fechas del trial actual
INICIO_MEDICION = date(2026, 11, 1)
FIN_PRUEBA = date(2027, 1, 31)

HITOS = [
    ("2026-10-30", "Primera decisión del bot (cierre del último día de bolsa de octubre)"),
    ("2026-10-31", "Tope para dejar escrita tu preferencia: protección en caídas o sencillez (antes de ver ningún resultado)"),
    ("2026-11-02", "Primeras órdenes simuladas, a la apertura"),
    ("2026-12-01", "Segunda tanda de órdenes"),
    ("2027-01-04", "Tercera y última tanda de órdenes de la prueba"),
    ("2027-01-31", "Fin de la prueba"),
    ("2027-02-01", "Decisión final: ¿seguir con el bot, pasar a cartera fija, o parar?"),
]

CRISIS_NOMBRES = {
    "2000": "Burbuja tecnológica",
    "2007": "Crisis financiera 2008",
    "2011": "Crisis deuda soberana",
    "2018": "Corrección 2018",
    "2020": "COVID-19",
    "2022": "Inflación / subida de tipos",
}

DIRECTORIO = Path(__file__).parent

# ---------------------------------------------------------------------------
# Descarga y preparación de precios
# ---------------------------------------------------------------------------

def descargar_precios() -> pd.DataFrame:
    """Descarga precios mensuales (último día de bolsa de cada mes) para los 4 ETFs."""
    frames = {}
    for activo, ticker in TICKERS.items():
        try:
            datos = yf.download(
                ticker,
                start=INICIO_BACKTEST,
                auto_adjust=True,
                progress=False,
            )
            if datos.empty:
                print(f"  AVISO: sin datos para {ticker}")
                continue
            # Colapsar MultiIndex si yfinance lo devuelve así
            if isinstance(datos.columns, pd.MultiIndex):
                datos.columns = datos.columns.get_level_values(0)
            # Resamplear al último día de bolsa de cada mes
            mensual = datos["Close"].resample("ME").last().dropna()
            frames[activo] = mensual
        except Exception as e:
            print(f"  ERROR descargando {ticker}: {e}")

    if not frames:
        raise RuntimeError("No se pudieron descargar precios para ningún ETF")

    df = pd.DataFrame(frames)
    df.index = df.index.normalize()  # quitar hora
    return df


def calcular_sma10(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula la SMA de 10 periodos mensuales."""
    return df.rolling(SMA_PERIODOS).mean()


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def _ultimo_dia_bolsa_mes(anio: int, mes: int, precios: pd.DataFrame) -> pd.Timestamp | None:
    """Devuelve el último día de cotización del mes dado en el índice de precios."""
    mask = (precios.index.year == anio) & (precios.index.month == mes)
    dias = precios.index[mask]
    return dias[-1] if len(dias) > 0 else None


def backtest(precios: pd.DataFrame, sma: pd.DataFrame) -> dict:
    """
    Simula el bot y la cartera fija desde INICIO_HISTORIA hasta hoy.
    Devuelve las estructuras necesarias para DATOS.historia.
    """
    activos = list(TICKERS.keys())

    # Filtrar desde inicio_historia en adelante (necesitamos sma válida)
    df = precios.loc[INICIO_HISTORIA:].copy()
    sm = sma.loc[INICIO_HISTORIA:].copy()

    # Excluir meses donde no haya sma para ningún activo
    df = df[~sm.isna().all(axis=1)]
    sm = sm.loc[df.index]

    meses = df.index.tolist()
    if not meses:
        return {}

    # Estado inicial
    cash_bot = CAPITAL_INICIAL
    posiciones = {a: 0.0 for a in activos}  # acciones
    valor_fija = CAPITAL_INICIAL

    curva_bot = []
    curva_fija = []
    retornos_bot = []    # rendimiento mensual del bot
    retornos_fija = []   # rendimiento mensual de la fija

    prev_bot = CAPITAL_INICIAL
    prev_fija = CAPITAL_INICIAL

    for i, mes in enumerate(meses):
        row = df.loc[mes]
        row_sma = sm.loc[mes]

        # Valor actual del portfolio bot antes de rebalancear
        val_posiciones = sum(posiciones[a] * row[a] for a in activos if a in row and not pd.isna(row[a]))
        val_bot = cash_bot + val_posiciones

        # Señales
        senales = {}
        for a in activos:
            if a not in row or pd.isna(row[a]) or a not in row_sma or pd.isna(row_sma[a]):
                senales[a] = None
            else:
                senales[a] = row[a] > row_sma[a]

        # Rebalanceo al final de mes
        objetivo = CAPITAL_INICIAL / len(activos)  # base, pero en realidad es val_bot / n_dentro
        activos_dentro = [a for a in activos if senales[a] is True]
        n_dentro = len(activos_dentro)

        # Vender activos que ya no están dentro
        for a in activos:
            if posiciones[a] > 0 and senales[a] is not True and a in row and not pd.isna(row[a]):
                importe = posiciones[a] * row[a] - COMISION
                cash_bot += importe
                posiciones[a] = 0.0

        # Comprar activos que están dentro
        if n_dentro > 0:
            asignacion = (cash_bot + sum(posiciones[a] * row[a] for a in activos_dentro if a in row and not pd.isna(row[a]))) / n_dentro
            for a in activos_dentro:
                if a not in row or pd.isna(row[a]):
                    continue
                precio_a = row[a]
                val_actual = posiciones[a] * precio_a
                diff = asignacion - val_actual
                if abs(diff) < precio_a * 0.5:  # cambio menor del 50% del precio: no rebalancear
                    continue
                if diff > 0:
                    acciones_comprar = int(diff / precio_a)
                    if acciones_comprar > 0:
                        coste = acciones_comprar * precio_a + COMISION
                        if coste <= cash_bot:
                            posiciones[a] += acciones_comprar
                            cash_bot -= coste
                else:
                    acciones_vender = int(abs(diff) / precio_a)
                    if acciones_vender > 0 and acciones_vender <= posiciones[a]:
                        posiciones[a] -= acciones_vender
                        cash_bot += acciones_vender * precio_a - COMISION

        # Valor final después de rebalanceo
        val_bot_final = cash_bot + sum(posiciones[a] * row[a] for a in activos if a in row and not pd.isna(row[a]))

        # Cartera fija (siempre 25% en cada activo, rebalanceo anual simplificado)
        if i == 0:
            # Distribución inicial
            pass
        else:
            prev_mes = meses[i - 1]
            # Rendimiento de la cartera fija: promedio ponderado de los 4 activos
            rendimientos = []
            for a in activos:
                if a in row and not pd.isna(row[a]) and a in df.loc[prev_mes] and not pd.isna(df.loc[prev_mes][a]):
                    r = row[a] / df.loc[prev_mes][a] - 1
                    rendimientos.append(r)
            if rendimientos:
                r_fija = sum(rendimientos) / len(rendimientos)
                valor_fija *= (1 + r_fija)

        fecha_str = mes.strftime("%Y-%m")
        curva_bot.append([fecha_str, round(val_bot_final, 2)])
        curva_fija.append([fecha_str, round(valor_fija, 2)])

        if i > 0:
            r_bot = val_bot_final / prev_bot - 1 if prev_bot > 0 else 0
            r_fija_m = valor_fija / prev_fija - 1 if prev_fija > 0 else 0
            retornos_bot.append(r_bot)
            retornos_fija.append(r_fija_m)

        prev_bot = val_bot_final
        prev_fija = valor_fija

    # Drawdown
    def calcular_drawdown(curva):
        picos = []
        caidas = []
        max_val = -math.inf
        for fecha_s, v in curva:
            if v > max_val:
                max_val = v
            caidas.append([fecha_s, round((v / max_val - 1), 6)])
        return caidas

    caida_bot = calcular_drawdown(curva_bot)
    caida_fija = calcular_drawdown(curva_fija)

    curva_combinada = [[c[0], c[1], f[1]] for c, f in zip(curva_bot, curva_fija)]
    caida_combinada = [[c[0], c[1], f[1]] for c, f in zip(caida_bot, caida_fija)]

    # Ventanas de 6 meses
    ventanas = []
    n = len(curva_bot)
    for i in range(n - 6):
        inicio_v = curva_bot[i][0]
        r_bot_6 = curva_bot[i + 6][1] / curva_bot[i][1] - 1
        r_fija_6 = curva_fija[i + 6][1] / curva_fija[i][1] - 1
        ventanas.append([inicio_v, round(r_bot_6, 6), round(r_fija_6, 6)])

    # Crisis: 5 peores periodos de la cartera fija de 6 meses
    peores = sorted(ventanas, key=lambda x: x[2])[:5]
    crisis = []
    for v in sorted(peores, key=lambda x: x[0]):
        # Encontrar el mes de fin (6 meses después)
        idx = next(i for i, c in enumerate(curva_fija) if c[0] == v[0])
        fin_idx = min(idx + 6, len(curva_fija) - 1)
        crisis.append({
            "inicio": v[0],
            "fin": curva_fija[fin_idx][0],
            "bot": round(curva_bot[fin_idx][1] / curva_bot[idx][1] - 1, 6),
            "fija": round(v[2], 6),
        })

    # Extremos
    vals_bot = [c[1] for c in curva_bot]
    vals_fija = [c[1] for c in curva_fija]

    def peor_caida_maxima(curva_vals):
        max_v = -math.inf
        max_dd = 0
        for v in curva_vals:
            if v > max_v:
                max_v = v
            dd = v / max_v - 1
            if dd < max_dd:
                max_dd = dd
        return round(max_dd, 6)

    # Peores días: no tenemos datos diarios en backtest; usamos meses como proxy
    peor_mes_bot_idx = min(range(len(retornos_bot)), key=lambda i: retornos_bot[i]) if retornos_bot else 0
    peor_mes_fija_idx = min(range(len(retornos_fija)), key=lambda i: retornos_fija[i]) if retornos_fija else 0

    def peor_ano(curva):
        anios = {}
        for i in range(1, len(curva)):
            anio = curva[i][0][:4]
            r = curva[i][1] / curva[i-1][1] - 1
            if anio not in anios:
                anios[anio] = 1.0
            anios[anio] *= (1 + r)
        if not anios:
            return ["2008", -0.1]
        peor_a = min(anios, key=lambda a: anios[a])
        return [peor_a, round(anios[peor_a] - 1, 6)]

    # Monte Carlo simple: reordenar retornos mensuales del bot
    mc_p95 = _monte_carlo(retornos_bot)

    extremos = {
        "bot": {
            "final": round(vals_bot[-1], 2),
            "caida_maxima": peor_caida_maxima(vals_bot),
            "peor_mes": [curva_bot[peor_mes_bot_idx + 1][0],
                         round(retornos_bot[peor_mes_bot_idx], 6)] if retornos_bot else ["2008-10", -0.15],
            "peor_ano": peor_ano(curva_bot),
        },
        "fija": {
            "final": round(vals_fija[-1], 2),
            "caida_maxima": peor_caida_maxima(vals_fija),
            "peor_mes": [curva_fija[peor_mes_fija_idx + 1][0],
                         round(retornos_fija[peor_mes_fija_idx], 6)] if retornos_fija else ["2008-10", -0.20],
            "peor_ano": peor_ano(curva_fija),
        },
        "mc_p95": round(mc_p95, 6),
    }

    # Ejemplo 2008 para la sección "Desde cero"
    ejemplo_2008 = _ejemplo_crisis_2008(precios, sma)

    return {
        "curva": curva_combinada,
        "caida": caida_combinada,
        "ventanas": ventanas,
        "crisis": crisis,
        "extremos": extremos,
        "eleccion": {"fecha": "2026-10-31", "opcion": "bot"},
        "_ejemplo_2008": ejemplo_2008,
    }


def _monte_carlo(retornos: list, n_sim: int = 5000, horizonte: int = 12) -> float:
    """Simulación Monte Carlo: peor resultado al 5% con 1.000 €, horizonte 12 meses."""
    if not retornos:
        return -0.20
    resultados = []
    for _ in range(n_sim):
        muestra = random.choices(retornos, k=horizonte)
        resultado = 1.0
        for r in muestra:
            resultado *= (1 + r)
        resultados.append(resultado - 1)
    resultados.sort()
    return resultados[int(0.05 * n_sim)]


def _ejemplo_crisis_2008(precios: pd.DataFrame, sma: pd.DataFrame) -> list:
    """Genera la serie del ETF de bolsa mundial 2007-2009 con señal del bot."""
    if "RV_MUNDIAL" not in precios.columns:
        return []
    try:
        mask = (precios.index >= "2007-01-01") & (precios.index <= "2009-12-31")
        sub = precios.loc[mask, "RV_MUNDIAL"].dropna()
        sub_sma = sma.loc[mask, "RV_MUNDIAL"].dropna()
        if sub.empty:
            return []
        # Normalizar a 100€ desde el primer mes
        base = sub.iloc[0]
        resultado = []
        for ts in sub.index:
            precio = sub[ts]
            media = sub_sma[ts] if ts in sub_sma.index and not pd.isna(sub_sma[ts]) else None
            dentro = bool(precio > media) if media is not None else None
            resultado.append([
                ts.strftime("%Y-%m"),
                round(precio / base * 100, 2),
                round(media / base * 100, 2) if media is not None else None,
                dentro,
            ])
        return resultado
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Prueba en seco (último año antes del trial)
# ---------------------------------------------------------------------------

def prueba_seco(precios: pd.DataFrame, sma: pd.DataFrame) -> dict:
    """Backtest del último año completo (12 meses anteriores al inicio del trial)."""
    fin_seco = INICIO_MEDICION - timedelta(days=1)
    inicio_seco = date(fin_seco.year - 1, fin_seco.month, 1)

    mask = (precios.index >= pd.Timestamp(inicio_seco)) & (precios.index <= pd.Timestamp(fin_seco))
    df_s = precios.loc[mask].copy()
    sm_s = sma.loc[mask].copy()

    if df_s.empty:
        return {"bot": [], "fija": [], "escalada": [], "operaciones": []}

    activos = list(TICKERS.keys())
    cash = CAPITAL_INICIAL
    posiciones = {a: 0.0 for a in activos}
    valor_fija = CAPITAL_INICIAL
    operaciones = []

    curva_bot = []
    curva_fija = []

    for i, mes in enumerate(df_s.index):
        row = df_s.loc[mes]
        row_sma = sm_s.loc[mes] if mes in sm_s.index else pd.Series(dtype=float)

        val_bot = cash + sum(posiciones[a] * row[a] for a in activos if a in row and not pd.isna(row[a]))

        senales = {}
        for a in activos:
            if a not in row or pd.isna(row[a]) or a not in row_sma.index or pd.isna(row_sma[a]):
                senales[a] = None
            else:
                senales[a] = bool(row[a] > row_sma[a])

        # Vender fuera
        for a in activos:
            if posiciones[a] > 0 and senales[a] is not True and a in row and not pd.isna(row[a]):
                importe = posiciones[a] * row[a] - COMISION
                operaciones.append({
                    "fecha": mes.strftime("%Y-%m-%d"),
                    "activo": a,
                    "lado": "venta",
                    "acciones": posiciones[a],
                    "precio": round(float(row[a]), 4),
                    "comision": COMISION,
                    "importe": round(importe, 2),
                })
                cash += importe
                posiciones[a] = 0.0

        # Comprar dentro
        activos_dentro = [a for a in activos if senales[a] is True]
        if activos_dentro:
            total_invertido = sum(posiciones[a] * row[a] for a in activos_dentro if a in row and not pd.isna(row[a]))
            asignacion = (cash + total_invertido) / len(activos_dentro)
            for a in activos_dentro:
                if a not in row or pd.isna(row[a]):
                    continue
                precio_a = float(row[a])
                val_actual = posiciones[a] * precio_a
                diff = asignacion - val_actual
                if diff > precio_a:
                    n = int(diff / precio_a)
                    if n > 0 and n * precio_a + COMISION <= cash:
                        operaciones.append({
                            "fecha": mes.strftime("%Y-%m-%d"),
                            "activo": a,
                            "lado": "compra",
                            "acciones": float(n),
                            "precio": round(precio_a, 4),
                            "comision": COMISION,
                            "importe": round(n * precio_a + COMISION, 2),
                        })
                        posiciones[a] += n
                        cash -= n * precio_a + COMISION

        val_bot_final = cash + sum(posiciones[a] * row[a] for a in activos if a in row and not pd.isna(row[a]))

        if i > 0:
            prev_mes = df_s.index[i - 1]
            rendimientos = []
            for a in activos:
                if a in row and not pd.isna(row[a]) and a in df_s.loc[prev_mes] and not pd.isna(df_s.loc[prev_mes][a]):
                    rendimientos.append(float(row[a]) / float(df_s.loc[prev_mes][a]) - 1)
            if rendimientos:
                valor_fija *= (1 + sum(rendimientos) / len(rendimientos))

        fecha_str = mes.strftime("%Y-%m")
        curva_bot.append([fecha_str, round(val_bot_final, 2)])
        curva_fija.append([fecha_str, round(valor_fija, 2)])

    # Escalada: cartera fija ajustada al nivel de riesgo del bot
    if curva_fija:
        escala = EXPOSICION_DEV
        base_fija = curva_fija[0][1]
        curva_escalada = [
            [c[0], round(CAPITAL_INICIAL + (c[1] - base_fija) * escala, 2)]
            for c in curva_fija
        ]
    else:
        curva_escalada = []

    return {
        "bot": curva_bot,
        "fija": curva_fija,
        "escalada": curva_escalada,
        "operaciones": operaciones,
    }


# ---------------------------------------------------------------------------
# Estado actual del trial
# ---------------------------------------------------------------------------

def cargar_estado() -> dict:
    """Carga estado.json o devuelve estado vacío."""
    ruta = DIRECTORIO / "estado.json"
    if ruta.exists():
        with open(ruta) as f:
            return json.load(f)
    return {
        "posiciones": {a: 0.0 for a in TICKERS},
        "caja": CAPITAL_INICIAL,
        "operaciones": [],
        "meses_procesados": [],
        "diario": [],
    }


def guardar_estado(estado: dict) -> None:
    ruta = DIRECTORIO / "estado.json"
    with open(ruta, "w") as f:
        json.dump(estado, f, ensure_ascii=False, indent=2)


def estado_actual(precios: pd.DataFrame, sma: pd.DataFrame) -> dict:
    """Calcula el estado del trial y genera datos actuales."""
    estado = cargar_estado()
    hoy = date.today()
    activos = list(TICKERS.keys())

    # Últimos precios disponibles
    ultima_fila = precios.iloc[-1]
    ultima_sma = sma.iloc[-1]
    fecha_datos = precios.index[-1].date()

    avisos = []
    if (hoy - fecha_datos).days > 3:
        avisos.append(f"precios del {fecha_datos.isoformat()}, pueden estar desactualizados")

    # Posiciones actuales con señales
    posiciones = []
    for a in activos:
        acciones = estado["posiciones"].get(a, 0.0)
        precio = float(ultima_fila[a]) if a in ultima_fila and not pd.isna(ultima_fila[a]) else 0.0
        valor_pos = acciones * precio

        senal = {}
        if a in ultima_fila and not pd.isna(ultima_fila[a]) and a in ultima_sma.index and not pd.isna(ultima_sma[a]):
            cierre = float(ultima_fila[a])
            media = float(ultima_sma[a])
            distancia = (cierre - media) / media
            senal = {
                "fecha": fecha_datos.isoformat(),
                "cierre": cierre,
                "media": media,
                "distancia": round(distancia, 8),
                "dentro": cierre > media,
            }

        posiciones.append({
            "activo": a,
            "etf": ETF_NOMBRES[a],
            "acciones": acciones,
            "precio": round(precio, 2),
            "valor": round(valor_pos, 2),
            "senal": senal,
        })

    # Valor total del portfolio
    valor_posiciones = sum(p["valor"] for p in posiciones)
    valor_total = estado["caja"] + valor_posiciones

    # Próxima decisión: último día de bolsa del mes actual
    proxima_decision = _proxima_decision(precios, hoy)

    # Rentabilidad del trial (solo desde inicio_medicion)
    rentabilidad = None
    if estado["meses_procesados"] and hoy >= INICIO_MEDICION:
        r_bot = valor_total / CAPITAL_INICIAL - 1
        # Rentabilidad de la cartera fija escalada (aproximación)
        r_fija = _rentabilidad_fija_desde(precios, INICIO_MEDICION)
        r_escalada = r_fija * EXPOSICION_DEV
        D = r_bot - r_escalada
        rentabilidad = {
            "bot": round(r_bot, 6),
            "escalada": round(r_escalada, 6),
            "D": round(D, 6),
        }

    # Series del trial (para el gráfico de "Hoy")
    series = _series_trial(precios, estado) if estado["meses_procesados"] else None

    # Actualizar diario
    # ETFs cerca de cambiar de señal (distancia < 3%)
    cerca = [
        p["activo"].replace("RV_MUNDIAL", "Bolsa mundial").replace("RV_EMERGENTE", "Emergentes")
        .replace("BONOS_EUR", "Bonos").replace("ORO", "Oro")
        + f" {p['senal']['distancia']:+.1%}"
        for p in posiciones
        if p["senal"] and abs(p["senal"].get("distancia", 1)) < 0.03
    ]
    cerca_str = ", ".join(cerca) if cerca else "ninguno"
    proxima = _proxima_decision(precios, hoy)
    entrada_hoy = f"| {hoy.isoformat()} | {fecha_datos.isoformat()} | {round(valor_total, 2)} | {estado['caja']:.2f} | — | {cerca_str} | {proxima} |"
    diario = estado.get("diario", [])
    if not diario or not diario[-1].startswith(f"| {hoy.isoformat()}"):
        diario.append(entrada_hoy)
        estado["diario"] = diario[-90:]  # últimas 90 entradas
        guardar_estado(estado)

    return {
        "posiciones": posiciones,
        "valor": round(valor_total, 2),
        "caja": round(estado["caja"], 2),
        "operaciones": estado.get("operaciones", []),
        "meses_procesados": estado.get("meses_procesados", []),
        "rentabilidad": rentabilidad,
        "series": series,
        "avisos": avisos,
        "fecha_datos": fecha_datos.isoformat(),
        "diario": estado.get("diario", []),
    }


def _proxima_decision(precios: pd.DataFrame, hoy: date) -> str:
    """Devuelve la fecha del último día de bolsa del mes actual (o siguiente)."""
    # Buscar el último día del mes en el índice de precios
    mes_actual = hoy.replace(day=1)
    # El último día del mes en el índice
    mask = (precios.index.year == hoy.year) & (precios.index.month == hoy.month)
    dias_mes = precios.index[mask]
    if len(dias_mes) > 0:
        return dias_mes[-1].strftime("%Y-%m-%d")
    # Si aún no hay datos del mes, estimar como el último día del mes anterior
    primer_dia_mes = pd.Timestamp(hoy.year, hoy.month, 1)
    ultimo_anterior = precios.index[precios.index < primer_dia_mes]
    if len(ultimo_anterior) > 0:
        return ultimo_anterior[-1].strftime("%Y-%m-%d")
    return hoy.isoformat()


def _rentabilidad_fija_desde(precios: pd.DataFrame, inicio: date) -> float:
    """Rentabilidad media de los 4 ETFs desde una fecha."""
    try:
        df = precios.loc[str(inicio):]
        if len(df) < 2:
            return 0.0
        r_total = []
        for a in TICKERS:
            if a in df.columns:
                vals = df[a].dropna()
                if len(vals) >= 2:
                    r_total.append(vals.iloc[-1] / vals.iloc[0] - 1)
        return sum(r_total) / len(r_total) if r_total else 0.0
    except Exception:
        return 0.0


def _series_trial(precios: pd.DataFrame, estado: dict) -> dict | None:
    """Genera las series del trial para el gráfico de Hoy."""
    try:
        inicio = pd.Timestamp(INICIO_MEDICION)
        df = precios.loc[inicio:]
        if df.empty:
            return None
        activos = list(TICKERS.keys())
        bot_vals = []
        fija_vals = []
        valor_fija = CAPITAL_INICIAL
        prev = None
        for mes in df.index:
            row = df.loc[mes]
            if prev is not None:
                rendimientos = []
                for a in activos:
                    if a in row and not pd.isna(row[a]) and a in df.loc[prev] and not pd.isna(df.loc[prev][a]):
                        rendimientos.append(float(row[a]) / float(df.loc[prev][a]) - 1)
                if rendimientos:
                    valor_fija *= (1 + sum(rendimientos) / len(rendimientos))
            mes_str = mes.strftime("%Y-%m")
            # Valor del bot desde estado actual
            bot_vals.append([mes_str, round(estado["caja"] + sum(
                estado["posiciones"].get(a, 0) * float(row[a])
                for a in activos if a in row and not pd.isna(row[a])
            ), 2)])
            fija_vals.append([mes_str, round(valor_fija, 2)])
            prev = mes
        if not bot_vals:
            return None
        escala = EXPOSICION_DEV
        base_f = fija_vals[0][1]
        escalada_vals = [[c[0], round(CAPITAL_INICIAL + (c[1] - base_f) * escala, 2)] for c in fija_vals]
        return {"bot": bot_vals, "fija": fija_vals, "escalada": escalada_vals}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Ensamblar DATOS
# ---------------------------------------------------------------------------

def generar_datos(precios: pd.DataFrame, sma: pd.DataFrame, historia: dict,
                  seco: dict, actual: dict) -> dict:
    """Ensambla el dict DATOS completo para inyectar en el HTML."""

    hoy = date.today()

    datos = {
        "generado": datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M+00:00"),
        "fecha_datos": actual["fecha_datos"],
        "inicio_medicion": INICIO_MEDICION.isoformat(),
        "proxima_decision": actual.get("proxima_decision",
                                       _proxima_decision(precios, hoy)),
        "capital_inicial": CAPITAL_INICIAL,
        "valor": actual["valor"],
        "caja": actual["caja"],
        "posiciones": actual["posiciones"],
        "umbral": UMBRAL,
        "exposicion_dev": EXPOSICION_DEV,
        "fin_prueba": FIN_PRUEBA.isoformat(),
        "decision_final": "febrero de 2027",
        "hitos": [{"fecha": f, "texto": t} for f, t in HITOS],
        "series": actual["series"],
        "operaciones": actual["operaciones"],
        "meses_procesados": actual["meses_procesados"],
        "rentabilidad": actual["rentabilidad"],
        "paridad": {"paridad_operaciones": None, "paridad_valor": None},
        "avisos": actual["avisos"],
        "historia": {
            "curva": historia.get("curva", []),
            "caida": historia.get("caida", []),
            "ventanas": historia.get("ventanas", []),
            "crisis": historia.get("crisis", []),
            "extremos": historia.get("extremos", {}),
            "eleccion": historia.get("eleccion", {"fecha": "2026-10-31", "opcion": "bot"}),
        },
        "ejemplo_2008": historia.get("_ejemplo_2008", []),
        "prueba_en_seco": seco,
        "diario": actual["diario"],
    }

    # Calcular proxima_decision correctamente
    datos["proxima_decision"] = _proxima_decision(precios, hoy)

    return datos


# ---------------------------------------------------------------------------
# Generar HTML
# ---------------------------------------------------------------------------

def generar_html(datos: dict) -> None:
    """Inyecta DATOS en la plantilla y escribe index.html."""
    plantilla_ruta = DIRECTORIO / "plantilla.html"
    salida_ruta = DIRECTORIO / "index.html"

    with open(plantilla_ruta, encoding="utf-8") as f:
        plantilla = f.read()

    datos_json = json.dumps(datos, ensure_ascii=False, separators=(",", ":"))
    html = plantilla.replace("/*DATOS_PLACEHOLDER*/", datos_json)

    with open(salida_ruta, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  index.html generado ({len(html):,} bytes)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Descargando precios...")
    precios = descargar_precios()
    print(f"  {len(precios)} meses, {precios.shape[1]} ETFs")
    print(f"  Desde {precios.index[0].date()} hasta {precios.index[-1].date()}")

    print("Calculando SMA 10 meses...")
    sma = calcular_sma10(precios)

    print("Ejecutando backtest histórico...")
    hist = backtest(precios, sma)

    print("Ejecutando prueba en seco (último año)...")
    seco = prueba_seco(precios, sma)

    print("Calculando estado actual del trial...")
    actual = estado_actual(precios, sma)

    print("Ensamblando DATOS...")
    datos = generar_datos(precios, sma, hist, seco, actual)

    print("Generando HTML...")
    generar_html(datos)

    print("Listo.")
    print(f"  Valor del bot: {datos['valor']} €")
    print(f"  Próxima decisión: {datos['proxima_decision']}")
    for p in datos["posiciones"]:
        s = p.get("senal", {})
        estado_str = "▲ Dentro" if s.get("dentro") else "▼ Fuera"
        dist = s.get("distancia", 0)
        print(f"  {p['activo']}: {estado_str} ({dist:+.1%})")


if __name__ == "__main__":
    main()
