# BR_lens_core.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Módulo Cython para el loop caliente del cálculo del polinomio de BR.

Compila con:
    cd scripts/
    python setup_BR.py build_ext --inplace

Reemplaza el Python puro en BR_lens_worker.py:
    - Todos los loops Python → loops C
    - bytearray → unsigned char* (memset para reset)
    - list[int]  → int[:] typed memoryview
    - Eliminadas todas las llamadas a funciones Python en el hot path

Speedup esperado vs Python puro: 5–15×
"""

import numpy as np
cimport numpy as cnp
from libc.string cimport memset

cnp.import_array()


def compute_BR_chunk(
    cnp.ndarray[cnp.int32_t, ndim=1] a_vals,
    int orb_sz,
    int p,
    cnp.ndarray[cnp.int32_t, ndim=1] phi,
    cnp.ndarray[cnp.int32_t, ndim=1] ed1,
    cnp.ndarray[cnp.int32_t, ndim=1] ed2,
    cnp.ndarray[cnp.int32_t, ndim=1] vertex_of,
    int m, int n, int V, int kG,
    cnp.ndarray[cnp.int64_t, ndim=2] binom,
):
    """
    Procesa un bloque de necklaces primitivos a ∈ a_vals con b ∈ [0, 2^p).
    Retorna dict {(ex, ey, ez): coeficiente}.

    Parámetros:
        a_vals   : array int32, necklaces primitivos del chunk
        orb_sz   : tamaño de la órbita (= p para primitivos)
        p        : primo
        phi      : array int32 de longitud n = 4p
        ed1, ed2 : arrays int32 de longitud m = 2p (darts de cada arista)
        vertex_of: array int32 de longitud n
        binom    : tabla de Pascal int64, (m+2) × (m+2)
    """
    cdef:
        int ai, i, j, idx, a, b
        int b_count = 1 << p
        long long mask   # necesita 2p bits; int32 desborda para p>=16
        int eA, kA, fA, gamma, s_exp, t_exp
        int d, start, nxt, x, root, r1, r2, xtmp
        long long c
        int n_a = a_vals.shape[0]

        # Arrays de trabajo locales — se alocan UNA vez por chunk
        cnp.int32_t[:]    vp         = np.empty(V, dtype=np.int32)
        cnp.uint8_t[:]    root_seen  = np.empty(V, dtype=np.uint8)
        cnp.uint8_t[:]    active     = np.empty(n, dtype=np.uint8)
        cnp.uint8_t[:]    vis        = np.empty(n, dtype=np.uint8)
        cnp.uint8_t[:]    av         = np.empty(V, dtype=np.uint8)

        # Typed memoryviews sobre los arrays de entrada
        cnp.int32_t[:] phi_mv  = phi
        cnp.int32_t[:] ed1_mv  = ed1
        cnp.int32_t[:] ed2_mv  = ed2
        cnp.int32_t[:] vof_mv  = vertex_of
        cnp.int64_t[:, :] bin_mv = binom
        cnp.int32_t[:] av_in   = a_vals

    coeffs = {}

    for ai in range(n_a):
        a = av_in[ai]
        for b in range(b_count):
            mask = <long long>a | (<long long>b << p)

            # ── Darts activos ─────────────────────────────────────────────
            memset(&active[0], 0, n)
            eA = 0
            for idx in range(m):
                if (mask >> idx) & 1:
                    active[ed1_mv[idx]] = 1
                    active[ed2_mv[idx]] = 1
                    eA += 1

            # ── Union-Find con compresión de camino ───────────────────────
            for i in range(V):
                vp[i] = i

            for idx in range(m):
                if (mask >> idx) & 1:
                    # find + compress r1
                    x = vof_mv[ed1_mv[idx]]
                    root = x
                    while vp[root] != root:
                        root = vp[root]
                    while vp[x] != root:
                        xtmp = vp[x]
                        vp[x] = root
                        x = xtmp
                    r1 = root
                    # find + compress r2
                    x = vof_mv[ed2_mv[idx]]
                    root = x
                    while vp[root] != root:
                        root = vp[root]
                    while vp[x] != root:
                        xtmp = vp[x]
                        vp[x] = root
                        x = xtmp
                    r2 = root
                    if r1 != r2:
                        vp[r1] = r2

            # Contar componentes conexas (sin Python set)
            memset(&root_seen[0], 0, V)
            kA = 0
            for i in range(V):
                x = i
                while vp[x] != x:
                    x = vp[x]
                if not root_seen[x]:
                    root_seen[x] = 1
                    kA += 1

            # ── Conteo de caras phi_H ─────────────────────────────────────
            memset(&vis[0], 0, n)
            memset(&av[0], 0, V)
            fA = 0
            for start in range(n):
                if not active[start]:
                    continue
                av[vof_mv[start]] = 1
                if vis[start]:
                    continue
                fA += 1
                d = start
                while not vis[d]:
                    vis[d] = 1
                    nxt = phi_mv[d] ^ 1          # sigma(d)
                    while not active[nxt]:
                        nxt = phi_mv[nxt] ^ 1    # buscar dart activo
                    d = nxt ^ 1                  # phi_H(d)
            for i in range(V):
                if not av[i]:
                    fA += 1

            # ── Acumulación de coeficientes ───────────────────────────────
            gamma = 2 * kA - V + eA - fA
            s_exp = kA - kG
            t_exp = eA - V + kA

            for j in range(s_exp + 1):
                c = bin_mv[s_exp, j] * orb_sz
                if (s_exp - j) & 1:
                    c = -c
                key = (j, t_exp, gamma)
                new_val = coeffs.get(key, 0) + c
                if new_val:
                    coeffs[key] = new_val
                elif key in coeffs:
                    del coeffs[key]

    return coeffs


def compute_BR_chunk_crest(
    cnp.ndarray[cnp.int32_t, ndim=1] a_vals,
    int p,
    cnp.ndarray[cnp.int32_t, ndim=1] phi,
    cnp.ndarray[cnp.int32_t, ndim=1] ed1,
    cnp.ndarray[cnp.int32_t, ndim=1] ed2,
    cnp.ndarray[cnp.int32_t, ndim=1] vertex_of,
    int m, int n, int V, int kG,
    bint strict,
):
    """
    Variante crest-only del loop caliente.

    Procesa a_vals con b ∈ [0, 2^p) y devuelve dict parcial {t: S_t_max}
    restringido a gamma=2, evitando la construcción del BR completo.
    """
    cdef:
        int ai, i, idx, a, b
        int b_count = 1 << p
        long long mask
        int eA, kA, fA, gamma, s_exp, t_exp
        int d, start, nxt, x, root, r1, r2, xtmp
        int n_a = a_vals.shape[0]

        cnp.int32_t[:] vp         = np.empty(V, dtype=np.int32)
        cnp.uint8_t[:] root_seen  = np.empty(V, dtype=np.uint8)
        cnp.uint8_t[:] active     = np.empty(n, dtype=np.uint8)
        cnp.uint8_t[:] vis        = np.empty(n, dtype=np.uint8)
        cnp.uint8_t[:] av         = np.empty(V, dtype=np.uint8)

        cnp.int32_t[:] phi_mv  = phi
        cnp.int32_t[:] ed1_mv  = ed1
        cnp.int32_t[:] ed2_mv  = ed2
        cnp.int32_t[:] vof_mv  = vertex_of
        cnp.int32_t[:] av_in   = a_vals

    best = {}

    for ai in range(n_a):
        a = av_in[ai]
        for b in range(b_count):
            mask = <long long>a | (<long long>b << p)

            memset(&active[0], 0, n)
            eA = 0
            for idx in range(m):
                if (mask >> idx) & 1:
                    active[ed1_mv[idx]] = 1
                    active[ed2_mv[idx]] = 1
                    eA += 1

            for i in range(V):
                vp[i] = i

            for idx in range(m):
                if (mask >> idx) & 1:
                    x = vof_mv[ed1_mv[idx]]
                    root = x
                    while vp[root] != root:
                        root = vp[root]
                    while vp[x] != root:
                        xtmp = vp[x]
                        vp[x] = root
                        x = xtmp
                    r1 = root

                    x = vof_mv[ed2_mv[idx]]
                    root = x
                    while vp[root] != root:
                        root = vp[root]
                    while vp[x] != root:
                        xtmp = vp[x]
                        vp[x] = root
                        x = xtmp
                    r2 = root

                    if r1 != r2:
                        vp[r1] = r2

            memset(&root_seen[0], 0, V)
            kA = 0
            for i in range(V):
                x = i
                while vp[x] != x:
                    x = vp[x]
                if not root_seen[x]:
                    root_seen[x] = 1
                    kA += 1

            memset(&vis[0], 0, n)
            memset(&av[0], 0, V)
            fA = 0
            for start in range(n):
                if not active[start]:
                    continue
                av[vof_mv[start]] = 1
                if vis[start]:
                    continue
                fA += 1
                d = start
                while not vis[d]:
                    vis[d] = 1
                    nxt = phi_mv[d] ^ 1
                    while not active[nxt]:
                        nxt = phi_mv[nxt] ^ 1
                    d = nxt ^ 1
            for i in range(V):
                if not av[i]:
                    fA += 1

            gamma = 2 * kA - V + eA - fA
            if strict and (gamma < 0 or (gamma & 1)):
                raise ValueError(
                    "Invariantes inconsistentes en Cython crest-only"
                )

            if gamma != 2:
                continue

            s_exp = kA - kG
            t_exp = eA - V + kA

            prev = best.get(t_exp)
            if prev is None or s_exp > prev:
                best[t_exp] = s_exp

    return best
