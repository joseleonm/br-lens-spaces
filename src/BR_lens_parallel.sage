#!/usr/bin/env sage
# -*- coding: utf-8 -*-
"""
BR_lens_parallel.sage
=====================
Versión paralela (multiprocessing + Cython) del cálculo del polinomio
de Bollobás-Riordan R(G_{p,q}; x,y,z) para los ribbon graphs de Heegaard
de los espacios lente L(p,q).

ARCHIVOS NECESARIOS (todos en el mismo directorio):
    BR_lens_parallel.sage   ← este script (orquestador Sage)
    BR_lens_worker.py       ← worker Python puro + dispatcher Cython
    BR_lens_core.pyx        ← módulo Cython (hot loop)
    setup_BR.py             ← script de compilación Cython

══════════════════════════════════════════════════════════════════════
INSTRUCCIONES DE INSTALACIÓN (Linux/macOS, una sola vez)
══════════════════════════════════════════════════════════════════════

1. INSTALAR SAGE (si no está):
       # Ubuntu/Debian:
       sudo apt install sagemath
       # Conda:
       conda install -c conda-forge sage

2. VERIFICAR DEPENDENCIAS (Cython y numpy deben estar en el entorno Sage):
       sage -c "import Cython; import numpy; print('OK')"

3. COMPILAR EL MÓDULO CYTHON (obligatorio, una vez por máquina):
       cd /ruta/al/directorio/
       sage -c "import subprocess, sys; subprocess.run([sys.executable, 'setup_BR.py', 'build_ext', '--inplace'], check=True)"
       # Alternativa directa:
       $(sage --python) setup_BR.py build_ext --inplace
       # Verificar que se generó el .so:
       ls BR_lens_core*.so

4. AJUSTAR P_LIST, N_WORKERS y OUT_JSON (ver sección CONFIGURACIÓN al final).

5. CORRER (puede tardar horas para p grandes; se recomienda screen/tmux):
       screen -S BR
       sage BR_lens_parallel.sage
       # Ctrl+A, D  para desacoplar

   O con nohup:
       nohup sage BR_lens_parallel.sage > BR.log 2>&1 &
       tail -f BR.log

══════════════════════════════════════════════════════════════════════
TIEMPOS APROXIMADOS (con Cython compilado, referencia orientativa)
══════════════════════════════════════════════════════════════════════

    p=13  (4 clases):  ~50s   en laptop 8 cores
                       ~20s   en servidor 24 cores

    p=17  (5 clases):  ~12 min  en servidor 12 cores
                       ~6 min   en servidor 24 cores

    p=19  (5 clases):  ~2 h   en servidor 12 cores
    p=20  (3 clases):  ~5 h   en servidor 12 cores

NOTA GPU: La GPU no ayuda. El cálculo es lógica discreta en CPU;
          el rendimiento escala linealmente con el número de cores.

══════════════════════════════════════════════════════════════════════
VERIFICACIÓN DE CORRECTITUD
══════════════════════════════════════════════════════════════════════
Para p ≤ 11 el script verifica automáticamente R(G;x,y,1) = T(G;x,y+1).
Para p=13,17 la verificación de Tutte es lenta; está desactivada por
defecto (VERIFICAR = False para p > 11). La correctitud fue validada
experimentalmente: los polinomios de clases homeomorfas coinciden y los
de clases no-homeomorfas difieren.

SALIDA:
    - Consola con progreso detallado
    - BR_polynomials.json  (ruta configurable vía OUT_JSON; se actualiza
      tras cada polinomio y hace MERGE con resultados previos)
"""

import time, json, os, sys
from datetime import datetime
from multiprocessing import Pool, cpu_count

# Agregar scripts/ al path para importar el worker
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import BR_lens_worker

# ─────────────────────────────────────────────────────────────────────────────
# Utilidades: ribbon graph de Heegaard, collares y clases de homeomorfismo
# ─────────────────────────────────────────────────────────────────────────────

def build_heegaard(p, q):
    assert gcd(p, q) == 1 and 1 <= q < p
    n = 4 * p
    def ao(i): return 2 * i
    def ai(i): return 2 * i + 1
    def bo(i): return 2 * p + 2 * i
    def bi(i): return 2 * p + 2 * i + 1

    sigma = [0] * n
    rho   = [0] * n
    for i in range(p):
        cyc = [ao(i), bo(i), ai((i - 1) % p), bi((i - q) % p)]
        for j in range(4):
            sigma[cyc[j]] = cyc[(j + 1) % 4]
    for i in range(p):
        rho[ao(i)] = ai(i);  rho[ai(i)] = ao(i)
        rho[bo(i)] = bi(i);  rho[bi(i)] = bo(i)

    phi = [rho[sigma[d]] for d in range(n)]
    edge_darts = [(ao(i), ai(i)) for i in range(p)] + \
                 [(bo(i), bi(i)) for i in range(p)]

    vertex_of = [-1] * n
    vis = bytearray(n)
    vi = 0
    for s in range(n):
        if not vis[s]:
            d = s
            while not vis[d]:
                vis[d] = 1
                vertex_of[d] = vi
                d = sigma[d]
            vi += 1
    return phi, edge_darts, vertex_of


def find_necklaces(p):
    mask_p = (1 << p) - 1
    result = []
    for a in range(1 << p):
        cur = a
        is_min = True
        for _ in range(p - 1):
            cur = ((cur << 1) | (cur >> (p - 1))) & mask_p
            if cur < a:
                is_min = False
                break
        if is_min:
            result.append(a)
    return result


def necklace_orbit_size(a, p):
    """Tamaño real de la órbita de a bajo rotación cíclica de longitud p.
    Para p primo siempre es p (salvo a=0 o a=all1 que es 1).
    Para p compuesto puede ser cualquier divisor de p."""
    mask_p = (1 << p) - 1
    cur = ((a << 1) | (a >> (p - 1))) & mask_p
    for k in range(1, p + 1):
        if cur == a:
            return k
        cur = ((cur << 1) | (cur >> (p - 1))) & mask_p
    return p  # no debería llegar aquí


def orbita(q, p):
    q_inv = power_mod(q, -1, p)
    return frozenset({q % p, (-q) % p, q_inv % p, (-q_inv) % p} & set(range(1, p)))


def representantes(p):
    seen = set()
    reps = []
    for q in range(1, p):
        if gcd(q, p) != 1 or q in seen:
            continue
        orb = orbita(q, p)
        seen |= orb
        reps.append(min(orb))
    return sorted(reps)


# ─────────────────────────────────────────────────────────────────────────────
# Cálculo paralelo del polinomio BR
# ─────────────────────────────────────────────────────────────────────────────

def BR_lens_parallel(p, q, n_workers=None, verbose=True):
    """
    Calcula R(G_{p,q}; x, y, z) usando multiprocessing.

    n_workers: número de procesos (default: cpu_count())
    """
    if n_workers is None:
        n_workers = cpu_count()

    phi, edge_darts, vertex_of = build_heegaard(p, q)
    n  = 4 * p
    m  = 2 * p
    V  = p
    kG = 1

    # Tabla de Pascal
    binom = [[0] * (m + 2) for _ in range(m + 2)]
    binom[0][0] = 1
    for ii in range(1, m + 2):
        binom[ii][0] = 1
        for jj in range(1, ii + 1):
            binom[ii][jj] = binom[ii - 1][jj - 1] + binom[ii - 1][jj]

    mask_p = (1 << p) - 1
    all1   = mask_p

    necklaces      = find_necklaces(p)
    prim_necklaces = [a for a in necklaces if a not in (0, all1)]

    n_canonical = 2 * len(necklaces) + len(prim_necklaces) * (1 << p)
    if verbose:
        print(f"  L({p},{q}): {n_canonical:,} máscaras canónicas | {n_workers} workers")

    t0 = time.time()
    coeffs = {}

    # ── Caso 1: a ∈ {0, all1}  →  b recorre collares (pequeño, secuencial) ──
    necklace_orb = {}
    for b in necklaces:
        if b == 0 or b == all1:
            necklace_orb[b] = 1
        else:
            necklace_orb[b] = necklace_orbit_size(b, p)  # correcto para p no primo

    args_special = (
        [0, all1], necklaces, necklace_orb,
        p, phi, edge_darts, vertex_of, m, n, V, kG, binom, all1
    )
    partial = BR_lens_worker.procesar_special(args_special)
    BR_lens_worker.merge_coeffs(coeffs, partial)

    # ── Caso 2: a collar primitivo → b ∈ [0, 2^p), paralelo ─────────────────
    # Agrupar por tamaño de órbita real (para p primo todos tienen orb_sz=p;
    # para p compuesto puede haber divisores propios de p).
    from collections import defaultdict
    by_orb_sz = defaultdict(list)
    for a in prim_necklaces:
        by_orb_sz[necklace_orbit_size(a, p)].append(a)

    worker_args = []
    for orb_sz, chunk_all in by_orb_sz.items():
        chunk_size = max(1, (len(chunk_all) + n_workers - 1) // n_workers)
        for i in range(0, len(chunk_all), chunk_size):
            chunk = chunk_all[i : i + chunk_size]
            # (a_chunk, orb_sz, p, phi, edge_darts, vertex_of, m, n, V, kG, binom)
            worker_args.append(
                (chunk, orb_sz, p, phi, edge_darts, vertex_of, m, n, V, kG, binom)
            )

    if verbose:
        print(f"  Distribuyendo {len(prim_necklaces)} necklaces en "
              f"{len(worker_args)} chunks ({len(by_orb_sz)} tamaños de órbita)...")

    with Pool(processes=n_workers) as pool:
        partials = pool.map(BR_lens_worker.procesar_chunk, worker_args)

    for partial in partials:
        BR_lens_worker.merge_coeffs(coeffs, partial)

    elapsed = time.time() - t0
    if verbose:
        throughput = n_canonical / elapsed
        print(f"  Completado en {elapsed:.2f}s  ({throughput:,.0f} máscaras/s)")

    # Construir polinomio Sage
    P = PolynomialRing(ZZ, 'x,y,z')
    x_v, y_v, z_v = P.gens()
    poly = P.zero()
    for (ex, ey, ez), c in coeffs.items():
        poly += c * x_v**ex * y_v**ey * z_v**ez

    return poly, elapsed


def verificar(p, q, poly, verbose=True):
    G = Graph(multiedges=True)
    G.add_vertices(range(p))
    for i in range(p):
        G.add_edge(i, (i + 1) % p, f'a{i}')
        G.add_edge(i, (i + q) % p, f'b{i}')
    T = G.tutte_polynomial()

    P_BR = poly.parent()
    x_br, y_br, z_br = P_BR.gens()
    R_z1 = poly.subs({z_br: 1})

    P_T = T.parent()
    x_t, y_t = P_T.gens()
    T_shifted = T.subs({y_t: y_t + 1})

    P2 = PolynomialRing(ZZ, ['x', 'y'])
    x2, y2 = P2.gens()
    try:
        ok = (P2(R_z1) == P2(T_shifted))
    except Exception:
        ok = (str(R_z1).replace(' ', '') == str(T_shifted).replace(' ', ''))

    if verbose:
        print(f"  Verificación Tutte: R(x,y,1) == T(x,y+1)  {'✓' if ok else '✗ FALLO'}")
    return ok


def guardar_web_json(results, outpath):
    """Guarda/actualiza BR_polynomials.json haciendo MERGE con casos existentes.

    Los casos en `results` reemplazan cualquier entrada (p,q) ya presente.
    Los casos existentes no incluidos en `results` se conservan intactos.
    Esto permite recalcular solo un subconjunto de p sin borrar los demás.
    """
    # Cargar casos existentes (si el archivo existe)
    existing = {}
    if os.path.exists(outpath):
        try:
            with open(outpath, 'r', encoding='utf-8') as f:
                old_data = json.load(f)
            for c in old_data.get('cases', []):
                existing[(c['p'], c['q'])] = c
        except Exception:
            pass  # Si el archivo está corrupto, empezar de cero

    # Construir casos nuevos desde results
    new_cases = {}
    for (p, q), (poly, orb, t) in sorted(results.items()):
        if not hasattr(poly, 'monomials'):
            continue
        terms = []
        for (ex, ey, ez), c in zip(
            [m.exponents()[0] for m in poly.monomials()],
            poly.coefficients()
        ):
            terms.append([int(ex), int(ey), int(ez), int(c)])
        terms.sort(key=lambda t: (-t[2], -t[1], -t[0]))
        max_ex = max((t[0] for t in terms), default=0)
        max_ey = max((t[1] for t in terms), default=0)
        max_ez = max((t[2] for t in terms), default=0)
        ez2 = [(t[0], t[1]) for t in terms if t[2] == 2 and t[3] != 0]
        crest_t, crest_S = [], []
        if ez2:
            ey_vals = sorted(set(ey for _, ey in ez2))
            for ey_val in ey_vals:
                s_val = max(ex for ex, ey in ez2 if ey == ey_val)
                crest_t.append(int(ey_val))
                crest_S.append(int(s_val))
        orb_sorted = sorted(int(v) for v in orb)
        new_cases[(int(p), int(q))] = {
            "p": int(p), "q": int(q),
            "label": f"L({p},{q})",
            "orbit": orb_sorted,
            "orbit_label": "{" + ",".join(str(v) for v in orb_sorted) + "}",
            "homeomorphism_class": " \u2245 ".join(f"L({p},{v})" for v in orb_sorted),
            "n_terms": len(terms),
            "max_ex": max_ex, "max_ey": max_ey, "max_ez": max_ez,
            "time_s": float(f"{float(t):.3f}"),
            "crest_profile": {"t": crest_t, "S": crest_S},
            "terms": terms,
        }

    # Merge: nuevos sobreescriben existentes; el resto se conserva
    existing.update(new_cases)
    cases = [v for _, v in sorted(existing.items())]

    dirpath = os.path.dirname(outpath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    data = {
        "generated": datetime.now().isoformat(),
        "description": "Bollobás-Riordan polynomials R(G_{p,q}; x,y,z) "
                       "for Heegaard ribbon graphs of lens spaces L(p,q). "
                       "Terms encoded as [ex, ey, ez, coeff].",
        "cases": cases,
    }
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(data, f, separators=(',', ':'), ensure_ascii=False)
    sz = float(os.path.getsize(outpath)) / 1024.0
    n_new = len(new_cases)
    print(f"Web JSON → {outpath}  ({sz:.1f} KB, {len(cases)} casos totales, {n_new} actualizados)")


# ─────────────────────────────────────────────────────────────────────────────
# Programa principal
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ══════════════════════════════════════════════════════════════════════
    # CONFIGURACIÓN — ajustar antes de correr
    # ══════════════════════════════════════════════════════════════════════
    #
    # P_LIST    : valores de p a calcular. Para p grande usar un servidor
    #             con muchos cores (p=19 ~2h, p=20 ~5h con 12 cores).
    # N_WORKERS : número de procesos paralelos. None = cpu_count() automático.
    # OUT_JSON  : ruta del archivo JSON de salida. Se hace MERGE con
    #             resultados previos; los casos no incluidos en P_LIST
    #             se conservan intactos.
    # ══════════════════════════════════════════════════════════════════════
    P_LIST    = [3, 5, 7, 11, 13]   # valores de p a calcular
    VERIFICAR = True                 # verifica R(x,y,1)=T(x,y+1) para p ≤ 11
    N_WORKERS = None                 # None = usar todos los cores disponibles
    OUT_JSON  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "data", "BR_polynomials.json")

    n_workers_actual = N_WORKERS if N_WORKERS is not None else cpu_count()
    print("=" * 65)
    print("  BR Paralelo — Espacios Lente L(p,q)")
    print(f"  Workers: {n_workers_actual}  |  cpu_count: {cpu_count()}")
    print("=" * 65)

    results    = {}
    total_time = 0.0

    for p in P_LIST:
        print(f"\n{'─'*65}")
        print(f"  p = {p}  |  clases: {representantes(p)}")
        print(f"{'─'*65}")

        for q in representantes(p):
            orb = orbita(q, p)
            nombres = " ≅ ".join(f"L({p},{v})" for v in sorted(orb))
            print(f"\n  Clase: {nombres}")

            poly, t = BR_lens_parallel(p, q, n_workers=n_workers_actual, verbose=True)
            results[(p, q)] = (poly, orb, t)
            total_time += t

            print(f"\n  R(G_{{{p},{q}}}; x,y,z) =\n  {poly}\n")

            if VERIFICAR and p <= 11:
                verificar(p, q, poly, verbose=True)

            guardar_web_json(results, OUT_JSON)

    print(f"\n{'='*65}")
    print(f"  {len(results)} polinomios  |  tiempo total: {total_time:.1f}s")
    print(f"{'='*65}\n")

    # Tabla resumen
    print(f"{'p':>3} {'q':>3} | {'términos':>8} | {'tiempo':>8} | Órbita")
    print("─" * 55)
    for (p, q), (poly, orb, t) in sorted(results.items()):
        try:
            n_terms = len(poly.monomials())
        except AttributeError:
            n_terms = "?"
        orb_str = "{" + ",".join(str(v) for v in sorted(orb)) + "}"
        print(f"{p:>3} {q:>3} | {n_terms!s:>8} | {t:>7.1f}s | {orb_str}")

    print("\nFin.")
