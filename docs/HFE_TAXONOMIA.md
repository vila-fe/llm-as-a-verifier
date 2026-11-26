# Glosario canónico HFE (Humanity's First Exam) — taxonomía oficial completa

Fuente: informe técnico de Google DeepMind (24-nov-2026),
[Humanity's First Exam: Measuring the Reliability Floor of LLMs](https://storage.googleapis.com/deepmind-media/papers/hfe_tech_paper.pdf).
Versión estructurada y verificada en `data/hfe_floor_trajs/hfe_taxonomy.json`
(sumas comprobadas: 20 subcategorías, 136 familias, 544 prompts).

## Estructura del benchmark
- **544 prompts** en **136 familias** de 4 prompts (1 semilla + 3 variantes que
  preservan la lógica, no necesariamente la respuesta).
- **228 prompts** (57 familias) text-only; **316 prompts** (79 familias)
  multimodales (texto e imagen); cada familia mantiene modalidad homogénea.
- Muestreo: **6 rollouts por prompt** (3.264 por modelo). Métrica principal de
  suelo: **family pass<sub>4</sub> (4×4)** — las 16 tiradas de la familia correctas.
- Partición pública: **112 prompts / 28 familias completas**, CC BY 4.0, en
  Kaggle (`deepmind/hfe-public`). Holdout privado: 432 prompts / 108 familias.

## Macro-categorías
| Macro | Familias | Prompts |
|---|---|---|
| Math & Algorithmic | 40 | 160 |
| Spatial & Positional | 27 | 108 |
| Vision | 51 | 204 |
| Reasoning & Language | 18 | 72 |

## Subcategorías (20)
| Subcategoría | Fam/Prompts | Mod. input | Macro |
|---|---|---|---|
| Algorithmic and Symbolic Execution | 8/32 | Text | Math & Alg. |
| Causal and Counterfactual Reasoning | 2/8 | Text(+Img) | R&L |
| Commonsense and Physical Reasoning | 4/16 | Text(+Img) | R&L |
| Constraint Satisfaction | 13/52 | Text | Math & Alg. |
| General Arithmetic and Calculation | 3/12 | Text | Math & Alg. |
| Generalization and Pattern Recognition | 4/16 | Text(+Img) | R&L |
| Logical Deduction | 2/8 | Text | R&L |
| Mathematical and Quantitative Reasoning | 5/20 | Text(+Img) | Math & Alg. |
| Positional Tracking | 6/24 | Text(+Img) | Spatial |
| Sequence Manipulation | 6/24 | Text | Math & Alg. |
| Sorting and Ordering | 5/20 | Text(+Img) | Math & Alg. |
| Spatial and Topological Reasoning | 4/16 | Text(+Img) | Spatial |
| Negative Fact Verification | 3/12 | Text(+Img) | R&L |
| Pragmatic Reasoning | 3/12 | Text | R&L |
| Color Identification | 10/40 | Text&Img | Vision |
| Object Counting | 12/48 | Text&Img | Vision |
| Text & Symbol Recognition (OCR) | 11/44 | Text&Img | Vision |
| Arrows, pointers, or indicators | 11/44 | Text&Img | Vision |
| Graph & Chart Understanding | 7/28 | Text&Img | Vision |
| Spatial Reasoning | 17/68 | Text(+Img) | Spatial |

Tres subcategorías piloto no sobrevivieron al filtrado de calidad y quedaron a
0 familias: Analogical Reasoning (Structural), Inductive and Probabilistic
Reasoning, Temporal Arithmetic.

## Nota para el mapeo de fricciones (Sofia)
El `id` de cada subcategoría en el JSON (p. ej. `constraint_satisfaction`,
`positional_tracking`, `negative_fact_verification`) coincide con la clave
`capability` de la suite Suelo HFE, de modo que los resultados HOOTL se
desglosan por subcategoría canónica directamente conectable a la taxonomía de
fricciones y sesgos del Marco Sofia.
