#!/usr/bin/env python3
"""Auditoria de efectividad del motor Suelo HFE (MacQAS Atomico / llm-as-a-verifier).

Auditoria tipo revisa-efectividad sobre el PIPELINE (no sobre un modelo): valida
que el motor de fiabilidad de suelo funciona correctamente usando datos reales
si estan disponibles y casos de frontera sinteticos declarados.

Criterios:
  A. Datos reales: carga el subconjunto real documentado (o la suite completa
     si fue ingerida) y verifica trazabilidad de respuestas de referencia.
  B. Efectividad del chequeo binario sobre prompts reales: answer_is_correct
     contra respuestas reales y variantes de formato/caso; se fallo = fallo.
  C. Logica de metricas y umbral: family-pass, cota Wilson, veredicto canonic
     (incluidos casos de frontera: todo-correcto, un solo fallo, n=28 familias).
  D. Integridad del registro de politica y determinismo HOOTL.

Salida: PASS/FAIL por criterio, veredicto global y reporte por consola. Con
--log, anexa el resultado al registro de politica (hfe_floor_log.jsonl).

Uso:
    python scripts/audit_hfe_suelo.py
    python scripts/audit_hfe_suelo.py --suite data/hfe_floor_trajs/hfe_public_floor.jsonl
"""

import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from llm_verifier.hfe_floor import (
    answer_is_correct,
    floor_metrics,
    load_suite,
    run_hootl,
    run_suite,
    wilson_lower_bound,
)

DEFAULT_REAL = os.path.join(ROOT_DIR, "data", "hfe_floor_trajs", "hfe_real_public_subset.jsonl")
LOG_PATH = os.path.join(ROOT_DIR, "data", "progress_tracking", "hfe_floor_log.jsonl")


def audit_a(trials):
    """Criterio A: datos reales cargados y trazables."""
    ok = len(trials) >= 1 and all(t.get("answer") and t.get("source") for t in trials)
    notes = [f"{len(trials)} prompts reales cargados",
             f"respuestas con referencia: {sum(1 for t in trials if t.get('answer'))}"]
    return ok, notes


def audit_b(trials):
    """Criterio B: chequeo binario efectivo sobre datos reales."""
    t0 = trials[0]
    cases = [
        (t0["answer"], True, "respuesta exacta aceptada"),
        (t0["answer"].upper(), True, "mayusculas no rompen el suelo"),
        (t0["answer"] + ".", True, "puntuacion final tolerada"),
        (" " + t0["answer"] + " ", True, "espacios tolerados"),
        (t0["answer"].replace("4", "5"), False, "conteo errado falla"),
        ("", False, "vacio falla"),
        ("I cannot see the image", False, "evasion falla"),
    ]
    fails = [(r, why) for r, exp, why in cases if answer_is_correct(r, t0["answer"]) != exp]
    for t in trials[1:]:
        if not answer_is_correct(t["answer"], t["answer"]):
            fails.append((t["answer"], "respuesta de referencia no reconocida"))
    notes = [f"{len(cases)} casos evaluados, {len(cases) - len(fails)} correctos"]
    for r, why in fails:
        notes.append(f"FALLO: {why} -> {r!r}")
    return not fails, notes


def audit_c():
    """Criterio C: metricas family-pass, Wilson y umbral canonic."""
    notes, ok = [], True
    # frontera: 1 familia de 4 prompts, todo correcto
    fam_run = (lambda correct: [
        {"id": f"p{i}", "family": "f", "capability": "c", "run": i, "response": "", "correct": correct[i]}
        for i in range(4)
    ])
    m = floor_metrics([{"id": f"p{i}", "family": "f"} for i in range(4)], fam_run([1, 1, 1, 1]))
    if abs(m["family_pass"] - 1.0) > 1e-9:
        ok = False; notes.append("family_pass != 1.0 con todo correcto")
    m = floor_metrics([{"id": f"p{i}", "family": "f"} for i in range(4)], fam_run([0, 1, 1, 1]))
    if m["family_pass"] != 0.0:
        ok = False; notes.append("family_pass != 0.0 con un solo fallo (la metrica estricta debe fallar la familia)")
    # Wilson: n=28 familias (particion publica completa), 28/28 -> cota >= 0.85 esperada
    lcb = wilson_lower_bound(28, 28)
    if lcb < 0.85:
        ok = False
        notes.append(f"cota Wilson con 28/28 familias ({lcb:.3f}) por debajo del umbral 0.85")
    if wilson_lower_bound(0, 0) != 0.0:
        ok = False; notes.append("Wilson con n=0 debe ser 0")
    if wilson_lower_bound(13, 28) >= 0.85:
        ok = False; notes.append("13/28 familias no deberia pasar el umbral")
    notes += [f"cota Wilson 28/28 familias: {lcb:.3f} (umbral 0.85: PASS esperado con dataset completo perfecto)",
              "regla: un solo fallo en 16 tiradas falla la familia (coherente con family pass4 4x4 del informe HFE)"]
    return ok, notes


def audit_d(trials, suite_path):
    """Criterio D: determinismo HOOTL y registro de politica."""
    notes, ok = [], True
    runs1 = run_suite(trials, {"object_counting": 0.9, "color_identification": 0.9}, 7)
    runs2 = run_suite(trials, {"object_counting": 0.9, "color_identification": 0.9}, 7)
    det = runs1 == runs2
    if not det:
        ok = False; notes.append("HOOTL no determinista con misma semilla")
    runs3 = run_suite(trials, {"object_counting": 0.9, "color_identification": 0.9}, 8)
    if runs3 == runs1:
        ok = False; notes.append("semillas distintas producen mismo resultado (sospecho de no variar)")
    notes.append(f"determinismo con misma semilla: {'OK' if det else 'FALLO'}")
    _, report = run_hootl(suite_path, seed=7, results_path=os.path.join(ROOT_DIR, "results", "audit_hfe_tmp.txt"))
    last = None
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, encoding="utf-8") as fh:
            lines = [l for l in fh if l.strip()]
        if lines:
            last = json.loads(lines[-1])
            notes.append(f"registro de politica activo: ultima entrada {last.get('module')}/{last.get('mode')} veredicto {last.get('verdict')}")
    if last is None:
        # la propia ejecucion de run_hootl sin --no-log ya debio anexar
        notes.append("aviso: registro de politica sin entradas antes de esta auditoria (se anexa la de ahora)")
    return ok, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=None, help="Suite real preferente (p. ej. hfe_public_floor.jsonl si existe).")
    parser.add_argument("--log", action="store_true", help=f"Anexar resultado al registro de politica {LOG_PATH}.")
    args = parser.parse_args()

    suite_path = args.suite or DEFAULT_REAL
    if not os.path.exists(suite_path) and args.suite is None:
        fallback = os.path.join(ROOT_DIR, "data", "hfe_floor_trajs", "hfe_public_floor.jsonl")
        suite_path = fallback if os.path.exists(fallback) else DEFAULT_REAL
    trials = load_suite(suite_path)
    print(f"Auditoria sobre: {os.path.basename(suite_path)}")
    print(f"  cobertura real del HFE publico (112 prompts): {len(trials)}/112")

    sections = [
        ("A. Datos reales y trazabilidad", audit_a(trials)),
        ("B. Chequeo binario sobre prompts reales", audit_b(trials)),
        ("C. Metricas y umbral canonicos", audit_c()),
        ("D. Determinismo HOOTL y registro", audit_d(trials, suite_path)),
    ]
    print()
    all_ok = True
    for name, (ok, notes) in sections:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        for n in notes:
            print(f"    - {n}")
        all_ok = all_ok and ok
    verdict = "PASS" if all_ok else "FAIL"
    print(f"\nVEREDICTO AUDITORIA DE EFECTIVIDAD: {verdict}")
    print("Nota: audita el MOTOR del suelo, no a ningun modelo. Evidencia de suelo")
    print("de un modelo requiere el dataset completo (HITL: zip de Kaggle).")

    if args.log and all_ok:
        entry = {
            "module": "hfe_floor",
            "mode": "AUDIT-EFECTIVIDAD",
            "timestamp": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
            "suite": os.path.basename(suite_path),
            "coverage_real_hfe_public": f"{len(trials)}/112",
            "criteria_pass": sum(1 for _, (ok, _) in sections if ok),
            "criteria_total": len(sections),
            "verdict": verdict,
        }
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"registro de politica: {LOG_PATH}")


if __name__ == "__main__":
    main()
