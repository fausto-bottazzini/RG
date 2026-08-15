import time
import numpy as np

from Metricas.minkowski import Minkowski
from Metricas.schwarzschild import Schwarzschild
from Metricas.kerr import Kerr

from Solver.geodesica import GeodesicaEvaluator
from Solver.solver import SolverGeodesica
from Solver.invariantes import Invariantes


# ============================================================
# CONFIGURACIÓN
# ============================================================

M = 1.0
A = 0.5

METODO = "DOP853"

RTOL = 1e-9
ATOL = 1e-11

RTOLS = (
    1e-6,
    1e-8,
    1e-9,
    1e-10,
)

ATOL_FACTOR = 1e-2

N_RHS = 100_000

TOL_TENSOR = 1e-10
TOL_INVARIANTE = 1e-8
TOL_REVERSIBILIDAD = 1e-7


# ============================================================
# UTILIDADES
# ============================================================

def titulo(texto):
    print()
    print("=" * 72)
    print(texto)
    print("=" * 72)


def subtitulo(texto):
    print()
    print("-" * 72)
    print(texto)
    print("-" * 72)


def medir(func, *args, **kwargs):
    inicio = time.perf_counter()
    resultado = func(*args, **kwargs)
    return resultado, time.perf_counter() - inicio


def estado(ok):
    return "PASS" if ok else "FAIL"


def max_abs(array):
    array = np.asarray(array, dtype=float)
    return float(np.max(np.abs(array)))


# ============================================================
# MÉTRICAS
# ============================================================

def construir_metricas():
    return {
        "Minkowski": Minkowski(),
        "Schwarzschild": Schwarzschild(M=M),
        "Kerr": Kerr(M=M, a=A),
    }


# ============================================================
# EVALUACIÓN NUMÉRICA DENSA
# ============================================================

def tensor_denso(tensor_numerico, coords, dim, rank):
    """
    Reconstruye un tensor denso únicamente para validación.

    El framework continúa trabajando internamente sparse.
    Esta función existe solamente para los tests.
    """

    out = np.zeros((dim,) * rank, dtype=float)

    valores = tensor_numerico.evaluar_valores(*coords)

    for idx, valor in zip(
        tensor_numerico.indices,
        valores,
    ):
        out[idx] = float(valor)

    return out


# ============================================================
# TEST DE MÉTRICA E INVERSA
# ============================================================

def test_metrica_inversa(metrica, coords):

    g = metrica.numeric("metric")
    gi = metrica.numeric("inverse_metric")

    G = tensor_denso(
        g,
        coords,
        metrica.g.dim,
        2,
    )

    GI = tensor_denso(
        gi,
        coords,
        metrica.g.dim,
        2,
    )

    identidad = G @ GI

    error = max_abs(
        identidad - np.eye(metrica.g.dim)
    )

    ok = error < TOL_TENSOR

    print(
        f"  g · g⁻¹ = I          : "
        f"{estado(ok)}   error = {error:.3e}"
    )

    return ok


# ============================================================
# TEST CHRISTOFFEL
# ============================================================

def test_christoffel(metrica, coords):

    Gamma_num = metrica.numeric(
        "christoffel"
    )

    Gamma = tensor_denso(
        Gamma_num,
        coords,
        metrica.g.dim,
        3,
    )

    # Gamma^rho_mu_nu = Gamma^rho_nu_mu
    error = max_abs(
        Gamma - np.swapaxes(
            Gamma,
            1,
            2,
        )
    )

    ok = error < TOL_TENSOR

    print(
        f"  simetría Christoffel : "
        f"{estado(ok)}   error = {error:.3e}"
    )

    return ok


# ============================================================
# TEST RIEMANN
# ============================================================

def test_riemann(metrica, coords):

    R_num = metrica.numeric("riemann")

    R = tensor_denso(
        R_num,
        coords,
        metrica.g.dim,
        4,
    )

    # R^rho_sigma_mu_nu =
    # -R^rho_sigma_nu_mu
    error_antisym = max_abs(
        R + np.swapaxes(
            R,
            2,
            3,
        )
    )

    # Identidad de Bianchi:
    #
    # R^rho_{ sigma mu nu}
    # + R^rho_{ sigma nu mu}
    # + R^rho_{ sigma ...}
    #
    # Para el chequeo completo de Bianchi necesitamos bajar
    # el primer índice. Lo hacemos abajo.

    g_num = metrica.numeric("metric")

    g = tensor_denso(
        g_num,
        coords,
        metrica.g.dim,
        2,
    )

    # R_{a b c d}
    R_cov = np.einsum(
        "ar,r b c d->a b c d",
        g,
        R,
    )

    # R_abcd = -R_bacd
    error_first = max_abs(
        R_cov + np.swapaxes(
            R_cov,
            0,
            1,
        )
    )

    # Bianchi algebraico:
    #
    # R_abcd + R_acdb + R_adbc = 0
    bianchi = (
        R_cov
        + np.transpose(
            R_cov,
            (0, 2, 3, 1),
        )
        + np.transpose(
            R_cov,
            (0, 3, 1, 2),
        )
    )

    error_bianchi = max_abs(
        bianchi
    )

    ok = (
        error_antisym < TOL_TENSOR
        and error_first < TOL_TENSOR
        and error_bianchi < TOL_TENSOR
    )

    print(
        f"  antisimetría Riemann : "
        f"{estado(error_antisym < TOL_TENSOR)}"
        f"   error = {error_antisym:.3e}"
    )

    print(
        f"  antisimetría primero : "
        f"{estado(error_first < TOL_TENSOR)}"
        f"   error = {error_first:.3e}"
    )

    print(
        f"  Bianchi              : "
        f"{estado(error_bianchi < TOL_TENSOR)}"
        f"   error = {error_bianchi:.3e}"
    )

    return ok


# ============================================================
# TEST RICCI / ESCALAR
# ============================================================

def test_curvatura(metrica, coords, esperado_vacio):

    Ric_num = metrica.numeric("ricci")

    Ric = tensor_denso(
        Ric_num,
        coords,
        metrica.g.dim,
        2,
    )

    error_ricci = max_abs(Ric)

    escalar = metrica.escalar_curvatura()

    # Para los espacios que deben ser Ricci-planos,
    # el escalar debe ser cero.
    if esperado_vacio:

        # El escalar ya existe simbólicamente.
        # Lo evaluamos mediante sustitución numérica.
        subs = dict(
            zip(
                metrica.coords,
                coords,
            )
        )

        scalar_value = float(
            escalar.subs(subs)
        )

        error_scalar = abs(
            scalar_value
        )

        ok = (
            error_ricci < TOL_TENSOR
            and error_scalar < TOL_TENSOR
        )

        print(
            f"  Ricci = 0            : "
            f"{estado(error_ricci < TOL_TENSOR)}"
            f"   error = {error_ricci:.3e}"
        )

        print(
            f"  escalar = 0          : "
            f"{estado(error_scalar < TOL_TENSOR)}"
            f"   error = {error_scalar:.3e}"
        )

        return ok

    return True


# ============================================================
# TEST COMPLETO DE GEOMETRÍA
# ============================================================

def test_geometria(nombre, metrica, coords):

    subtitulo(
        f"GEOMETRÍA - {nombre}"
    )

    print(
        f"  dimensión             : "
        f"{metrica.g.dim}"
    )

    print(
        f"  g nnz                 : "
        f"{metrica.g.nnz}"
    )

    print(
        f"  g⁻¹ nnz               : "
        f"{metrica.g_inv.nnz}"
    )

    Gamma = metrica.christoffel()

    print(
        f"  Γ nnz                 : "
        f"{Gamma.nnz}"
    )

    R = metrica.riemann()

    print(
        f"  Riemann nnz           : "
        f"{R.nnz}"
    )

    Ric = metrica.ricci()

    print(
        f"  Ricci nnz             : "
        f"{Ric.nnz}"
    )

    ok = True

    ok &= test_metrica_inversa(
        metrica,
        coords,
    )

    ok &= test_christoffel(
        metrica,
        coords,
    )

    ok &= test_riemann(
        metrica,
        coords,
    )

    if nombre in (
        "Minkowski",
        "Schwarzschild",
        "Kerr",
    ):
        ok &= test_curvatura(
            metrica,
            coords,
            esperado_vacio=True,
        )

    return ok


# ============================================================
# BENCHMARK CONSTRUCCIÓN
# ============================================================

def benchmark_construccion():

    titulo(
        "BENCHMARK DE CONSTRUCCIÓN Y CACHÉ"
    )

    resultados = {}

    for nombre, constructor in (
        (
            "Minkowski",
            lambda: Minkowski(),
        ),
        (
            "Schwarzschild",
            lambda: Schwarzschild(M=M),
        ),
        (
            "Kerr",
            lambda: Kerr(M=M, a=A),
        ),
    ):

        print()
        print(nombre)

        metrica, t_metric = medir(
            constructor
        )

        print(
            f"  métrica               : "
            f"{t_metric:.6f} s"
        )

        tiempos = {}

        for cantidad in (
            "metric",
            "inverse_metric",
            "derivadas",
            "christoffel",
            "derivadas_christoffel",
            "riemann",
            "ricci",
        ):

            try:

                _, tiempo = medir(
                    metrica.numeric,
                    cantidad,
                )

                tiempos[cantidad] = tiempo

                print(
                    f"  {cantidad:<22}: "
                    f"{tiempo:.6f} s"
                )

            except Exception as exc:

                tiempos[cantidad] = None

                print(
                    f"  {cantidad:<22}: "
                    f"SKIP ({type(exc).__name__})"
                )

        # Segunda llamada: debe usar caché.
        _, tiempo_cache = medir(
            metrica.numeric,
            "christoffel",
        )

        print(
            f"  Christoffel cache     : "
            f"{tiempo_cache:.6e} s"
        )

        resultados[nombre] = {
            "metrica": t_metric,
            "tiempos": tiempos,
            "cache": tiempo_cache,
        }

    return resultados


# ============================================================
# BENCHMARK RHS
# ============================================================

def benchmark_rhs(nombre, metrica, y):

    solver = SolverGeodesica(metrica)

    evaluator = solver.evaluator

    out = np.empty(
        2 * evaluator.dim,
        dtype=float,
    )

    # Warm-up.
    for _ in range(1000):
        evaluator.sistema(
            y,
            out,
        )

    inicio = time.perf_counter()

    for _ in range(N_RHS):
        evaluator.sistema(
            y,
            out,
        )

    tiempo = time.perf_counter() - inicio

    rhs_s = N_RHS / tiempo

    print()
    print(nombre)

    print(
        f"  evaluaciones          : "
        f"{N_RHS:,}"
    )

    print(
        f"  tiempo                : "
        f"{tiempo:.6f} s"
    )

    print(
        f"  RHS / segundo         : "
        f"{rhs_s:,.0f}"
    )

    print(
        f"  tiempo / RHS         : "
        f"{tiempo / N_RHS:.3e} s"
    )

    return rhs_s


# ============================================================
# CASOS GEODÉSICOS
# ============================================================

def caso_schwarzschild_circular():

    r = 10.0

    f = 1.0 - 2.0 * M / r

    E = f / np.sqrt(
        1.0 - 3.0 * M / r
    )

    L = np.sqrt(M * r) / np.sqrt(
        1.0 - 3.0 * M / r
    )

    x0 = np.array([
        0.0,
        r,
        np.pi / 2,
        0.0,
    ])

    u0 = np.array([
        E / f,
        0.0,
        0.0,
        L / r**2,
    ])

    return x0, u0, (E, L)


def caso_schwarzschild_general():

    x0 = np.array([
        0.0,
        10.0,
        np.pi / 2,
        0.0,
    ])

    u0 = np.array([
        1.2,
        -0.05,
        0.0,
        0.08,
    ])

    return x0, u0


def caso_kerr_ecuatorial():

    x0 = np.array([
        0.0,
        10.0,
        np.pi / 2,
        0.0,
    ])

    u0 = np.array([
        1.2,
        -0.05,
        0.0,
        0.08,
    ])

    return x0, u0


def caso_kerr_inclinada():

    x0 = np.array([
        0.0,
        10.0,
        np.pi / 2,
        0.0,
    ])

    u0 = np.array([
        1.2,
        -0.05,
        0.02,
        0.08,
    ])

    return x0, u0


# ============================================================
# CHEQUEO DE INVARIANTES
# ============================================================

def test_invariantes(
    nombre,
    metrica,
    resultado,
):

    inv = Invariantes(metrica)

    errores = inv.errores(
        resultado
    )

    print()
    print(
        f"INVARIANTES - {nombre}"
    )

    ok = True

    for key, label in (
        ("norma", "norma"),
        ("energia", "energía"),
        ("momento_angular", "Lz"),
        ("carter", "Carter Q"),
    ):

        valor = errores[key]

        if not np.isfinite(
            valor["max_relativo"]
        ):
            continue

        error = valor[
            "max_relativo"
        ]

        passed = error < TOL_INVARIANTE

        print(
            f"  {label:<18}: "
            f"{estado(passed)}"
            f"   abs = "
            f"{valor['max_absoluto']:.3e}"
            f"   rel = "
            f"{error:.3e}"
        )

        ok &= passed

    return ok


# ============================================================
# TEST ÓRBITA CIRCULAR
# ============================================================

def test_orbita_circular():

    subtitulo(
        "Schwarzschild - órbita circular"
    )

    metric = Schwarzschild(M=M)

    x0, u0, (E_ref, L_ref) = (
        caso_schwarzschild_circular()
    )

    solver = SolverGeodesica(
        metric
    )

    resultado = solver.resolver(
        x0,
        u0,
        (0.0, 100.0),
        metodo=METODO,
        rtol=RTOL,
        atol=ATOL,
    )

    if not resultado.success:

        print(
            "  integración          : FAIL"
        )

        return False

    inv = Invariantes(metric)

    datos = inv.trayectoria(
        resultado
    )

    r_error = max_abs(
        resultado.y[1] - x0[1]
    )

    theta_error = max_abs(
        resultado.y[2] - np.pi / 2
    )

    E_error = max_abs(
        datos["energia"] - E_ref
    )

    L_error = max_abs(
        datos["momento_angular"] - L_ref
    )

    ok = (
        r_error < 1e-7
        and theta_error < 1e-8
        and E_error < 1e-8
        and L_error < 1e-8
    )

    print(
        f"  integración          : PASS"
    )

    print(
        f"  radio constante      : "
        f"{estado(r_error < 1e-7)}"
        f"   error = {r_error:.3e}"
    )

    print(
        f"  plano ecuatorial     : "
        f"{estado(theta_error < 1e-8)}"
        f"   error = {theta_error:.3e}"
    )

    print(
        f"  energía analítica    : "
        f"{estado(E_error < 1e-8)}"
        f"   error = {E_error:.3e}"
    )

    print(
        f"  momento analítico    : "
        f"{estado(L_error < 1e-8)}"
        f"   error = {L_error:.3e}"
    )

    return ok


# ============================================================
# TEST GEODÉSICA GENERAL
# ============================================================

def test_geodesica(
    nombre,
    metrica,
    x0,
    u0,
    intervalo,
):

    subtitulo(nombre)

    solver = SolverGeodesica(
        metrica
    )

    inicio = time.perf_counter()

    resultado = solver.resolver(
        x0,
        u0,
        intervalo,
        metodo=METODO,
        rtol=RTOL,
        atol=ATOL,
    )

    tiempo = (
        time.perf_counter()
        - inicio
    )

    rhs_s = (
        resultado.nfev / tiempo
        if tiempo > 0
        else np.nan
    )

    print(
        f"  éxito                 : "
        f"{estado(resultado.success)}"
    )

    print(
        f"  tiempo                : "
        f"{tiempo:.6f} s"
    )

    print(
        f"  nfev                  : "
        f"{resultado.nfev}"
    )

    print(
        f"  pasos                 : "
        f"{len(resultado.t)}"
    )

    print(
        f"  RHS / segundo         : "
        f"{rhs_s:,.0f}"
    )

    if not resultado.success:
        return False, resultado

    invariantes_ok = test_invariantes(
        nombre,
        metrica,
        resultado,
    )

    return invariantes_ok, resultado


# ============================================================
# REVERSIBILIDAD
# ============================================================

def test_reversibilidad(
    nombre,
    metrica,
    x0,
    u0,
    intervalo=(0.0, 50.0),
):

    subtitulo(
        f"Reversibilidad - {nombre}"
    )

    solver = SolverGeodesica(
        metrica
    )

    forward = solver.resolver(
        x0,
        u0,
        intervalo,
        metodo=METODO,
        rtol=RTOL,
        atol=ATOL,
    )

    if not forward.success:
        print("  forward              : FAIL")
        return False

    yf = forward.y[:, -1]

    xf = yf[:4]
    uf = yf[4:]

    backward = solver.resolver(
        xf,
        -uf,
        (0.0, intervalo[1]),
        metodo=METODO,
        rtol=RTOL,
        atol=ATOL,
    )

    if not backward.success:
        print("  backward             : FAIL")
        return False

    recovered = backward.y[:, -1]

    target = np.concatenate(
        (
            x0,
            -u0,
        )
    )

    error = max_abs(
        recovered - target
    )

    ok = (
        error
        < TOL_REVERSIBILIDAD
    )

    print(
        f"  error máximo         : "
        f"{error:.3e}"
    )

    print(
        f"  reversibilidad       : "
        f"{estado(ok)}"
    )

    return ok


# ============================================================
# CONVERGENCIA
# ============================================================

def test_convergencia(
    nombre,
    metrica,
    x0,
    u0,
    intervalo,
):

    subtitulo(
        f"Convergencia - {nombre}"
    )

    solver = SolverGeodesica(
        metrica
    )

    resultados = []

    # Referencia mucho más precisa.
    referencia = solver.resolver(
        x0,
        u0,
        intervalo,
        metodo=METODO,
        rtol=1e-12,
        atol=1e-14,
    )

    if not referencia.success:
        print(
            "  referencia            : FAIL"
        )
        return False

    y_ref = referencia.y[:, -1]

    print()
    print(
        f"{'rtol':>10}"
        f"{'atol':>12}"
        f"{'nfev':>10}"
        f"{'pasos':>10}"
        f"{'tiempo':>12}"
        f"{'error estado':>16}"
    )

    print("-" * 72)

    ok = True

    for rtol in RTOLS:

        atol = (
            rtol
            * ATOL_FACTOR
        )

        inicio = time.perf_counter()

        resultado = solver.resolver(
            x0,
            u0,
            intervalo,
            metodo=METODO,
            rtol=rtol,
            atol=atol,
        )

        tiempo = (
            time.perf_counter()
            - inicio
        )

        y = resultado.y[:, -1]

        escala = np.maximum(
            np.abs(y_ref),
            1.0,
        )

        error = max_abs(
            (y - y_ref) / escala
        )

        print(
            f"{rtol:10.1e}"
            f"{atol:12.1e}"
            f"{resultado.nfev:10d}"
            f"{len(resultado.t):10d}"
            f"{tiempo:12.6f}"
            f"{error:16.3e}"
        )

        resultados.append(
            error
        )

    # No exigimos una ley exacta porque DOP853
    # y problemas distintos pueden no mostrar
    # orden ideal en todo el rango.
    #
    # Exigimos solamente que aumentar precisión
    # no destruya sistemáticamente la precisión.
    for i in range(
        1,
        len(resultados),
    ):

        if (
            resultados[i]
            > resultados[i - 1]
            * 1.5
        ):
            ok = False

    print()
    print(
        f"  comportamiento         : "
        f"{estado(ok)}"
    )

    return ok


# ============================================================
# TEST DE CACHÉ
# ============================================================

def test_cache():

    subtitulo(
        "Caché simbólico y numérico"
    )

    metric = Kerr(
        M=M,
        a=A,
    )

    # Primera construcción.
    _, t1 = medir(
        metric.christoffel
    )

    # Segunda llamada.
    _, t2 = medir(
        metric.christoffel
    )

    # Primera conversión numérica.
    _, t3 = medir(
        metric.numeric,
        "christoffel",
    )

    # Segunda conversión.
    _, t4 = medir(
        metric.numeric,
        "christoffel",
    )

    print(
        f"  Christoffel 1         : "
        f"{t1:.6f} s"
    )

    print(
        f"  Christoffel cache     : "
        f"{t2:.6e} s"
    )

    print(
        f"  numérico 1            : "
        f"{t3:.6f} s"
    )

    print(
        f"  numérico cache        : "
        f"{t4:.6e} s"
    )

    # No imponemos una relación rígida porque depende
    # del sistema de archivos y de SymPy.
    ok = (
        t2 <= t1
        and t4 <= t3
    )

    print(
        f"  reutilización          : "
        f"{estado(ok)}"
    )

    return ok


# ============================================================
# MAIN
# ============================================================

def main():

    titulo(
        "PRUEBA INTEGRAL DEL FRAMEWORK RG"
    )

    print()
    print("Configuración")
    print(f"  M                     = {M}")
    print(f"  a                     = {A}")
    print(f"  método                = {METODO}")
    print(f"  rtol                  = {RTOL}")
    print(f"  atol                  = {ATOL}")

    resultados = {}

    # ========================================================
    # CONSTRUCCIÓN
    # ========================================================

    benchmark_construccion()

    # ========================================================
    # GEOMETRÍA
    # ========================================================

    metricas = construir_metricas()

    coordenadas = {
        "Minkowski": np.array([
            0.3,
            2.0,
            1.0,
            0.7,
        ]),

        "Schwarzschild": np.array([
            0.3,
            10.0,
            1.1,
            0.7,
        ]),

        "Kerr": np.array([
            0.3,
            10.0,
            1.1,
            0.7,
        ]),
    }

    resultados["geometria"] = {}

    for nombre, metrica in metricas.items():

        resultados["geometria"][nombre] = (
            test_geometria(
                nombre,
                metrica,
                coordenadas[nombre],
            )
        )

    # ========================================================
    # RHS
    # ========================================================

    titulo(
        "BENCHMARK RHS"
    )

    y_test = np.array([
        0.0,
        10.0,
        np.pi / 2,
        0.0,
        1.2,
        -0.05,
        0.02,
        0.08,
    ])

    resultados["rhs"] = {}

    for nombre in (
        "Schwarzschild",
        "Kerr",
    ):

        resultados["rhs"][nombre] = (
            benchmark_rhs(
                nombre,
                metricas[nombre],
                y_test,
            )
        )

    # ========================================================
    # GEODÉSICAS
    # ========================================================

    titulo(
        "VALIDACIÓN DE GEODÉSICAS"
    )

    # --------------------------------------------------------
    # Schwarzschild radial
    # --------------------------------------------------------

    metric = metricas["Schwarzschild"]

    x0 = np.array([
        0.0,
        10.0,
        np.pi / 2,
        0.0,
    ])

    u0 = np.array([
        1.0,
        -0.1,
        0.0,
        0.0,
    ])

    ok, _ = test_geodesica(
        "Schwarzschild radial",
        metric,
        x0,
        u0,
        (0.0, 50.0),
    )

    resultados["radial"] = ok

    # --------------------------------------------------------
    # Schwarzschild general
    # --------------------------------------------------------

    x0, u0 = (
        caso_schwarzschild_general()
    )

    ok, _ = test_geodesica(
        "Schwarzschild general",
        metric,
        x0,
        u0,
        (0.0, 100.0),
    )

    resultados["schwarzschild_general"] = ok

    # --------------------------------------------------------
    # Schwarzschild circular
    # --------------------------------------------------------

    resultados["circular"] = (
        test_orbita_circular()
    )

    # --------------------------------------------------------
    # Kerr ecuatorial
    # --------------------------------------------------------

    metric = metricas["Kerr"]

    x0, u0 = (
        caso_kerr_ecuatorial()
    )

    ok, resultado_kerr = (
        test_geodesica(
            "Kerr ecuatorial",
            metric,
            x0,
            u0,
            (0.0, 100.0),
        )
    )

    resultados["kerr_ecuatorial"] = ok

    # --------------------------------------------------------
    # Kerr inclinada
    # --------------------------------------------------------

    x0, u0 = (
        caso_kerr_inclinada()
    )

    ok, resultado_kerr_inclinada = (
        test_geodesica(
            "Kerr inclinada",
            metric,
            x0,
            u0,
            (0.0, 100.0),
        )
    )

    resultados["kerr_inclinada"] = ok

    # ========================================================
    # KERR CARTER EXPLÍCITO
    # ========================================================

    subtitulo(
        "Kerr - chequeo explícito de Carter"
    )

    inv = Invariantes(
        metric
    )

    datos = inv.trayectoria(
        resultado_kerr_inclinada
    )

    Q0 = datos["carter"][0]

    Q_error = max_abs(
        datos["carter"] - Q0
    )

    print(
        f"  Q inicial             : "
        f"{Q0:.12e}"
    )

    print(
        f"  error máximo Q        : "
        f"{Q_error:.3e}"
    )

    resultados["carter"] = (
        Q_error
        < TOL_INVARIANTE
    )

    # ========================================================
    # REVERSIBILIDAD
    # ========================================================

    titulo(
        "REVERSIBILIDAD"
    )

    x0, u0 = (
        caso_schwarzschild_general()
    )

    resultados["reversibilidad_schwarzschild"] = (
        test_reversibilidad(
            "Schwarzschild",
            metricas["Schwarzschild"],
            x0,
            u0,
        )
    )

    x0, u0 = (
        caso_kerr_inclinada()
    )

    resultados["reversibilidad_kerr"] = (
        test_reversibilidad(
            "Kerr",
            metricas["Kerr"],
            x0,
            u0,
        )
    )

    # ========================================================
    # CONVERGENCIA
    # ========================================================

    titulo(
        "CONVERGENCIA NUMÉRICA"
    )

    x0, u0 = (
        caso_schwarzschild_general()
    )

    resultados["convergencia_schwarzschild"] = (
        test_convergencia(
            "Schwarzschild",
            metricas["Schwarzschild"],
            x0,
            u0,
            (0.0, 100.0),
        )
    )

    x0, u0 = (
        caso_kerr_inclinada()
    )

    resultados["convergencia_kerr"] = (
        test_convergencia(
            "Kerr",
            metricas["Kerr"],
            x0,
            u0,
            (0.0, 100.0),
        )
    )

    # ========================================================
    # CACHÉ
    # ========================================================

    resultados["cache"] = (
        test_cache()
    )

    # ========================================================
    # RESUMEN
    # ========================================================

    titulo(
        "RESULTADO FINAL"
    )

    for nombre, resultado in resultados.items():

        if isinstance(
            resultado,
            dict,
        ):
            continue

        print(
            f"  {nombre:<32}: "
            f"{estado(resultado)}"
        )

    tests_booleanos = []

    def recoger(obj):

        if isinstance(
            obj,
            bool,
        ):
            tests_booleanos.append(
                obj
            )

        elif isinstance(
            obj,
            dict,
        ):

            for value in obj.values():
                recoger(value)

    recoger(resultados)

    global_ok = all(
        tests_booleanos
    )

    print()
    print("=" * 72)

    if global_ok:
        print(
            "FRAMEWORK RG: VALIDACIÓN COMPLETA PASS"
        )
    else:
        print(
            "FRAMEWORK RG: HAY TESTS FALLIDOS"
        )

    print("=" * 72)

    return global_ok


if __name__ == "__main__":
    main()