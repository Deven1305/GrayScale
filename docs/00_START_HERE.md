# START HERE — documentation reading order

Files are numbered. Read them in order; each assumes the ones before it.
You do not need all of them — the table says who each one is for.

| # | Document | Read it if you want to | ~time |
|---|---|---|---|
| **01** | `01_SETUP.md` | Install the environment and verify the GPU build | 10 min |
| **02** | `02_RUN_WITHOUT_GPU.md` | **No NVIDIA GPU?** CPU + Colab + Kaggle instructions | 10 min |
| **03** | `03_HOW_TO_RUN.md` | **Run anything.** Every command, what `.npy` is, is-it-working checks | 20 min |
| **04** | `04_DATASET.md` | Understand the data before touching the model | 15 min |
| **05** | `05_FORENSICS_REPORT.md` | See how the degradation was reverse-engineered, with evidence | 40 min |
| **06** | `06_TECHNICAL_ARCHITECTURE.md` | Understand the model, pipeline and every design choice | 30 min |
| **07** | `07_ANALYSIS.md` | See EDA -> training -> benchmarks -> where it fails | 25 min |
| **08** | `08_OOD_REPORT.md` | See per-family generalisation results | 10 min |
| **09** | `09_ABLATION_RESULTS.md` | See which loss terms actually earned their place | 10 min |
| **10** | `10_IMPROVEMENTS.md` | Know the weaknesses and how to validate without ground truth | 20 min |
| **11** | `11_FUTURE_SCOPE.md` | Pick up the next piece of work | 20 min |
| **12** | `12_WHY_NOT_AN_LLM.md` | Know why this is a 15 M CNN and not a fine-tuned LLM/VLM | 25 min |

## If you only have 20 minutes

Read the root `README.md`, then **03** to run it, then **07 section 5** for the
honest list of what the model gets wrong.

## By role

| You are | Read |
|---|---|
| A reviewer checking it runs | root `README.md`, then **03** |
| A teammate without a GPU | **02**, then **03** |
| A teammate picking up development | **01**, **03**, **04**, **06** |
| Someone auditing the method | **05**, **07**, **08**, **09** |
| Someone continuing the research | **10**, **11** |
| Asking "why not just fine-tune an LLM?" | **12** |

Numbers, tables and figures throughout come from
`../experiments/EXPERIMENT_LOG.md`; nothing in documents **01-11** is estimated.
**12** is the one exception — it compares against systems we did not build, so
every figure there is tagged measured / derived / published / estimated.
