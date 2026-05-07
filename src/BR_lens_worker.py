"""
BR_lens_worker.py
=================
Worker para el cálculo paralelo del polinomio de Bollobás-Riordan.

Usa automáticamente el módulo Cython (BR_lens_core) si está compilado,
cayendo de vuelta a Python puro si no está disponible.

Para compilar el módulo Cython:
    cd scripts/
    python setup_BR.py build_ext --inplace   (o el python3 de Sage)

Usado por BR_lens_parallel.sage vía multiprocessing.Pool.
"""

import os, sys
import numpy as np

# Intentar cargar el módulo Cython compilado
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import BR_lens_core as _cy
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

# ─────────────────────────────────────────────────────────────────────────────
# Entrada principal del worker (picklable, usada por Pool.map)
# ─────────────────────────────────────────────────────────────────────────────

def procesar_chunk(args):
    """Dispatcher: usa Cython si está disponible, si no Python puro."""
    if HAS_CYTHON:
        return _procesar_chunk_cy(args)
    return _procesar_chunk_py(args)


def procesar_chunk_crest(args):
    """
    Dispatcher crest-only: devuelve perfil parcial {t: S_t_max}.

    Evita construir coeficientes BR completos y trabaja directo sobre
    subgrafos con gamma=2.
    """
    if HAS_CYTHON:
        return _procesar_chunk_crest_cy(args)
    return _procesar_chunk_crest_py(args)


def _procesar_chunk_cy(args):
    """
    Versión Cython: convierte listas a arrays numpy y llama a BR_lens_core.
    """
    a_chunk, orb_sz, p, phi, edge_darts, vertex_of, m, n, V, kG, binom = args

    # Convertir a arrays numpy tipados (requerido por las typed memoryviews)
    a_arr  = np.array(a_chunk,   dtype=np.int32)
    phi_ar = np.array(phi,       dtype=np.int32)
    ed1    = np.array([e[0] for e in edge_darts], dtype=np.int32)
    ed2    = np.array([e[1] for e in edge_darts], dtype=np.int32)
    vof    = np.array(vertex_of, dtype=np.int32)
    bin_ar = np.array(binom,     dtype=np.int64)

    return _cy.compute_BR_chunk(
        a_arr, orb_sz, p, phi_ar, ed1, ed2, vof, m, n, V, kG, bin_ar
    )


def _procesar_chunk_crest_cy(args):
    """
    Versión Cython crest-only.
    args = (a_chunk, orb_sz, p, phi, edge_darts, vertex_of, m, n, V, kG, strict)
    Retorna: dict parcial {t: S_t_max}
    """
    a_chunk, _orb_sz, p, phi, edge_darts, vertex_of, m, n, V, kG, strict = args

    a_arr  = np.array(a_chunk, dtype=np.int32)
    phi_ar = np.array(phi, dtype=np.int32)
    ed1    = np.array([e[0] for e in edge_darts], dtype=np.int32)
    ed2    = np.array([e[1] for e in edge_darts], dtype=np.int32)
    vof    = np.array(vertex_of, dtype=np.int32)

    return _cy.compute_BR_chunk_crest(
        a_arr, p, phi_ar, ed1, ed2, vof, m, n, V, kG, bool(strict)
    )


def _procesar_chunk_py(args):
    """
    Versión Python puro (fallback si Cython no está compilado).
    args = (a_chunk, orb_sz, p, phi, edge_darts, vertex_of, m, n, V, kG, binom)
    Retorna: dict parcial {(ex, ey, ez): coef_entero}
    """
    a_chunk, orb_sz, p, phi, edge_darts, vertex_of, m, n, V, kG, binom = args

    # Arrays locales por worker (sin compartir estado con otros procesos)
    active        = bytearray(n)
    vis           = bytearray(n)
    active_vertex = bytearray(V)
    vp            = list(range(V))
    vp_init       = list(range(V))
    zeros_n       = bytearray(n)
    zeros_V       = bytearray(V)

    coeffs = {}
    b_count = 1 << p

    for a in a_chunk:
        for b in range(b_count):
            mask = a | (b << p)

            # ── Darts activos ───────────────────────────────────────────────
            active[:] = zeros_n
            eA = 0
            for idx in range(m):
                if (mask >> idx) & 1:
                    d1, d2 = edge_darts[idx]
                    active[d1] = 1
                    active[d2] = 1
                    eA += 1

            # ── Union-Find inline ───────────────────────────────────────────
            vp[:] = vp_init
            for idx in range(m):
                if (mask >> idx) & 1:
                    x = vertex_of[edge_darts[idx][0]]
                    root = x
                    while vp[root] != root:
                        root = vp[root]
                    while vp[x] != root:
                        vp[x], x = root, vp[x]
                    r1 = root

                    x = vertex_of[edge_darts[idx][1]]
                    root = x
                    while vp[root] != root:
                        root = vp[root]
                    while vp[x] != root:
                        vp[x], x = root, vp[x]
                    r2 = root

                    if r1 != r2:
                        vp[r1] = r2

            roots = set()
            for i in range(V):
                x = i
                while vp[x] != x:
                    x = vp[x]
                roots.add(x)
            kA = len(roots)

            # ── Conteo de caras (phi_H) ─────────────────────────────────────
            vis[:] = zeros_n
            active_vertex[:] = zeros_V
            fA = 0
            for start in range(n):
                if not active[start]:
                    continue
                active_vertex[vertex_of[start]] = 1
                if vis[start]:
                    continue
                fA += 1
                d = start
                while not vis[d]:
                    vis[d] = 1
                    nxt = phi[d] ^ 1        # sigma(d) = rho(phi(d)) = phi(d) XOR 1
                    while not active[nxt]:
                        nxt = phi[nxt] ^ 1  # avanzar hasta dart activo
                    d = nxt ^ 1             # phi_H(d) = rho(sigma_H(d))
            # Vértices aislados → 1 cara de frontera cada uno
            for i in range(V):
                if not active_vertex[i]:
                    fA += 1

            # ── Acumulación ─────────────────────────────────────────────────
            gamma = 2 * kA - V + eA - fA   # = 2 * genus, ∈ {0, 2}
            s_exp = kA - kG
            t_exp = eA - V + kA

            binom_s = binom[s_exp]
            for j in range(s_exp + 1):
                c = binom_s[j] * orb_sz
                if (s_exp - j) & 1:
                    c = -c
                key = (j, t_exp, gamma)
                new_val = coeffs.get(key, 0) + c
                if new_val:
                    coeffs[key] = new_val
                elif key in coeffs:
                    del coeffs[key]

    return coeffs


def _procesar_chunk_crest_py(args):
    """
    Versión Python puro crest-only (fallback).
    args = (a_chunk, orb_sz, p, phi, edge_darts, vertex_of, m, n, V, kG, strict)
    Retorna: dict parcial {t: S_t_max}
    """
    a_chunk, _orb_sz, p, phi, edge_darts, vertex_of, m, n, V, kG, strict = args

    active        = bytearray(n)
    vis           = bytearray(n)
    active_vertex = bytearray(V)
    vp            = list(range(V))
    vp_init       = list(range(V))
    zeros_n       = bytearray(n)
    zeros_V       = bytearray(V)

    best = {}
    b_count = 1 << p

    for a in a_chunk:
        for b in range(b_count):
            mask = a | (b << p)

            active[:] = zeros_n
            eA = 0
            for idx in range(m):
                if (mask >> idx) & 1:
                    d1, d2 = edge_darts[idx]
                    active[d1] = 1
                    active[d2] = 1
                    eA += 1

            vp[:] = vp_init
            for idx in range(m):
                if (mask >> idx) & 1:
                    x = vertex_of[edge_darts[idx][0]]
                    root = x
                    while vp[root] != root:
                        root = vp[root]
                    while vp[x] != root:
                        vp[x], x = root, vp[x]
                    r1 = root

                    x = vertex_of[edge_darts[idx][1]]
                    root = x
                    while vp[root] != root:
                        root = vp[root]
                    while vp[x] != root:
                        vp[x], x = root, vp[x]
                    r2 = root

                    if r1 != r2:
                        vp[r1] = r2

            roots = set()
            for i in range(V):
                x = i
                while vp[x] != x:
                    x = vp[x]
                roots.add(x)
            kA = len(roots)

            vis[:] = zeros_n
            active_vertex[:] = zeros_V
            fA = 0
            for start in range(n):
                if not active[start]:
                    continue
                active_vertex[vertex_of[start]] = 1
                if vis[start]:
                    continue
                fA += 1
                d = start
                while not vis[d]:
                    vis[d] = 1
                    nxt = phi[d] ^ 1
                    while not active[nxt]:
                        nxt = phi[nxt] ^ 1
                    d = nxt ^ 1
            for i in range(V):
                if not active_vertex[i]:
                    fA += 1

            gamma = 2 * kA - V + eA - fA
            if strict and (gamma < 0 or (gamma & 1)):
                raise ValueError(
                    "Invariantes inconsistentes en worker crest-only: "
                    f"gamma={gamma}, mask={mask:#x}, kA={kA}, eA={eA}, fA={fA}"
                )

            if gamma != 2:
                continue

            s_exp = kA - kG
            t_exp = eA - V + kA
            old = best.get(t_exp)
            if old is None or s_exp > old:
                best[t_exp] = s_exp

    return best


def procesar_special(args):
    """
    Caso especial: a ∈ {0, all1}, b recorre los collares (necklaces).

    args = (a_list, necklaces, necklace_orb, p, phi, edge_darts, vertex_of,
            m, n, V, kG, binom, all1)

    necklace_orb: dict {b: orb_sz} — tamaño de la órbita de b.
    Retorna: dict parcial {(ex, ey, ez): coef_entero}
    """
    a_list, necklaces, necklace_orb, p, phi, edge_darts, vertex_of, \
        m, n, V, kG, binom, all1 = args

    active        = bytearray(n)
    vis           = bytearray(n)
    active_vertex = bytearray(V)
    vp            = list(range(V))
    vp_init       = list(range(V))
    zeros_n       = bytearray(n)
    zeros_V       = bytearray(V)

    coeffs = {}

    for a in a_list:
        for b in necklaces:
            mask = a | (b << p)
            orb_sz = necklace_orb[b]

            active[:] = zeros_n
            eA = 0
            for idx in range(m):
                if (mask >> idx) & 1:
                    d1, d2 = edge_darts[idx]
                    active[d1] = 1
                    active[d2] = 1
                    eA += 1

            vp[:] = vp_init
            for idx in range(m):
                if (mask >> idx) & 1:
                    x = vertex_of[edge_darts[idx][0]]
                    root = x
                    while vp[root] != root:
                        root = vp[root]
                    while vp[x] != root:
                        vp[x], x = root, vp[x]
                    r1 = root

                    x = vertex_of[edge_darts[idx][1]]
                    root = x
                    while vp[root] != root:
                        root = vp[root]
                    while vp[x] != root:
                        vp[x], x = root, vp[x]
                    r2 = root

                    if r1 != r2:
                        vp[r1] = r2

            roots = set()
            for i in range(V):
                x = i
                while vp[x] != x:
                    x = vp[x]
                roots.add(x)
            kA = len(roots)

            vis[:] = zeros_n
            active_vertex[:] = zeros_V
            fA = 0
            for start in range(n):
                if not active[start]:
                    continue
                active_vertex[vertex_of[start]] = 1
                if vis[start]:
                    continue
                fA += 1
                d = start
                while not vis[d]:
                    vis[d] = 1
                    nxt = phi[d] ^ 1
                    while not active[nxt]:
                        nxt = phi[nxt] ^ 1
                    d = nxt ^ 1
            for i in range(V):
                if not active_vertex[i]:
                    fA += 1

            gamma = 2 * kA - V + eA - fA
            s_exp = kA - kG
            t_exp = eA - V + kA

            binom_s = binom[s_exp]
            for j in range(s_exp + 1):
                c = binom_s[j] * orb_sz
                if (s_exp - j) & 1:
                    c = -c
                key = (j, t_exp, gamma)
                new_val = coeffs.get(key, 0) + c
                if new_val:
                    coeffs[key] = new_val
                elif key in coeffs:
                    del coeffs[key]

    return coeffs


def procesar_special_crest(args):
    """
    Caso especial crest-only: a ∈ {0, all1}, b recorre solo collares.

    args = (a_list, necklaces, p, phi, edge_darts, vertex_of, m, n, V, kG, strict)
    Retorna: dict parcial {t: S_t_max}
    """
    a_list, necklaces, p, phi, edge_darts, vertex_of, m, n, V, kG, strict = args

    active        = bytearray(n)
    vis           = bytearray(n)
    active_vertex = bytearray(V)
    vp            = list(range(V))
    vp_init       = list(range(V))
    zeros_n       = bytearray(n)
    zeros_V       = bytearray(V)

    best = {}

    for a in a_list:
        for b in necklaces:
            mask = a | (b << p)

            active[:] = zeros_n
            eA = 0
            for idx in range(m):
                if (mask >> idx) & 1:
                    d1, d2 = edge_darts[idx]
                    active[d1] = 1
                    active[d2] = 1
                    eA += 1

            vp[:] = vp_init
            for idx in range(m):
                if (mask >> idx) & 1:
                    x = vertex_of[edge_darts[idx][0]]
                    root = x
                    while vp[root] != root:
                        root = vp[root]
                    while vp[x] != root:
                        vp[x], x = root, vp[x]
                    r1 = root

                    x = vertex_of[edge_darts[idx][1]]
                    root = x
                    while vp[root] != root:
                        root = vp[root]
                    while vp[x] != root:
                        vp[x], x = root, vp[x]
                    r2 = root

                    if r1 != r2:
                        vp[r1] = r2

            roots = set()
            for i in range(V):
                x = i
                while vp[x] != x:
                    x = vp[x]
                roots.add(x)
            kA = len(roots)

            vis[:] = zeros_n
            active_vertex[:] = zeros_V
            fA = 0
            for start in range(n):
                if not active[start]:
                    continue
                active_vertex[vertex_of[start]] = 1
                if vis[start]:
                    continue
                fA += 1
                d = start
                while not vis[d]:
                    vis[d] = 1
                    nxt = phi[d] ^ 1
                    while not active[nxt]:
                        nxt = phi[nxt] ^ 1
                    d = nxt ^ 1
            for i in range(V):
                if not active_vertex[i]:
                    fA += 1

            gamma = 2 * kA - V + eA - fA
            if strict and (gamma < 0 or (gamma & 1)):
                raise ValueError(
                    "Invariantes inconsistentes en special crest-only: "
                    f"gamma={gamma}, mask={mask:#x}, kA={kA}, eA={eA}, fA={fA}"
                )

            if gamma != 2:
                continue

            s_exp = kA - kG
            t_exp = eA - V + kA
            old = best.get(t_exp)
            if old is None or s_exp > old:
                best[t_exp] = s_exp

    return best


def merge_coeffs(base, partial):
    """Acumula el dict parcial en base (in-place)."""
    for key, c in partial.items():
        new_val = base.get(key, 0) + c
        if new_val:
            base[key] = new_val
        elif key in base:
            del base[key]


def merge_crest(base, partial):
    """Acumula un perfil crest parcial {t: S_t_max} en base (in-place)."""
    for t_exp, s_exp in partial.items():
        old = base.get(t_exp)
        if old is None or s_exp > old:
            base[t_exp] = s_exp
