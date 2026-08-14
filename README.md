# Synthetic Dataset Generation and Screening — Demo

A self-contained, runnable demonstration of the synthetic dataset generation and
screening workflow described in the manuscript pseudocode. The repository ships
one notebook plus a small `raw_data/` directory; it does **not** depend on any
unreleased production scripts.

The workflow operates on **four receiver traces of 128 samples each**, taken from
real labeled acoustic/array data, and produces synthetic augmented training
samples through temporal translation, differential moveout, and four
augmentation configurations — followed by two physical screening steps.

---

## Repository contents

```
.
├── synthetic-workflow-demo.ipynb   # The full workflow, with intermediate results plotted
├── raw_data/
│   ├── inputs/                     # input_00000000.dat … input_00000008.dat  (9 samples)
│   └── labels/                     # label_00000000.dat … label_00000008.dat  (9 samples)
├── requirements.txt
└── README.md
```

---

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

The notebook requires Python 3.8+ and the packages listed in `requirements.txt`.

### 2. Launch the notebook

```bash
jupyter notebook synthetic-workflow-demo.ipynb
```

Then run all cells top to bottom. The notebook locates the `raw_data/` directory
automatically (it searches the current directory and its parents), so place the
notebook and `raw_data/` in the same directory — the layout this repository
already provides.

### 3. Experiment

Change the parameters in the configuration cell and re-run:

| Parameter | Description | Range / default |
|---|---|---|
| `SEED` | RNG seed for reproducible augmentation | `42` |
| `SAMPLE_ID` | Which released sample to load (`0`–`8`) | `0` |
| `DEPTH_INDEX` | Depth point within the 64-depth window to visualize | `31` |
| `GLOBAL_DIRECTION` | Global translation direction (`"left"` / `"right"`) | `"right"` |
| `GLOBAL_SHIFT_AMOUNT` | Translation in integer samples | `8` (left `1–47`, right `1–40`) |
| `MOVEOUT_DIRECTION` | Moveout direction (`"left"` / `"right"`) | `"right"` |
| `MOVEOUT_STEP` | Moveout step `n` (R2–R4 shift by `0.5n, n, 1.5n`) | `2` (left `1–7`, right `1–12`) |
| `DELTA_MAX` | Final screening threshold `δ_max` in samples | `12.0` |

---

## Data format

Each sample is a 64-depth sliding window extracted from a real labeled record.

### Inputs — `raw_data/inputs/input_XXXXXXXX.dat`

Little-endian `float32` array of shape `(64, 1, 128, 4)`:

- `64` — depth points,
- `1` — channel,
- `128` — time samples per trace,
- `4` — receivers R1–R4.

### Labels — `raw_data/labels/label_XXXXXXXX.dat`

Little-endian `float32` array with two concatenated parts:

1. **One-hot mask** — the first `64 × 1 × 128 × 4` values, reshaped to
   `(64, 1, 128, 4)`, with a `1.0` at the first-arrival sample index of each
   trace.
2. **Arrival-time picks** — the remaining `64 × 4` values, reshaped to `(64, 4)`,
   normalized to `[0, 1]`. Multiply by `127` to recover the sample index.

The loading logic (embedded in the notebook) is:

```python
def load_dat_sample(data_dir, sample_id):
    w = np.fromfile(data_dir / "inputs" / f"input_{sample_id:08d}.dat",
                    dtype=np.float32).reshape(64, 1, 128, 4)
    label = np.fromfile(data_dir / "labels" / f"label_{sample_id:08d}.dat",
                        dtype=np.float32)
    mask_size = 64 * 1 * 128 * 4
    m = label[:mask_size].reshape(64, 1, 128, 4)
    p = label[mask_size:].reshape(64, 4)
    return w, m, p
```

---

## Workflow

The notebook executes five stages:

1. **Load** one 64-depth sample and select the four receiver traces at one depth
   point.
2. **Synthesize** two mechanisms:
   - **Global Temporal Translation** — left shift truncates the left boundary and
     extrapolates the right boundary using the CWT-derived dominant frequency,
     Hilbert phase, five harmonics, and historical-value smoothing; a right shift
     zero-initializes the newly exposed left-boundary samples.
   - **Differential Moveout Simulation** — R1 is held fixed while R2–R4 are
     shifted by `0.5n`, `n`, and `1.5n`. The implementation performs 2×
     interpolation, CWT dominant-frequency estimation, boundary construction,
     resampling to 128 samples, smoothing of the first five samples, and
     three-point parabolic refinement of the picks.
3. **First screening** — compute adjacent inter-receiver arrival-time differences
   `Δ_i = t_{i+1} - t_i`; discard the sample if any `Δ_i < 0` or
   `|Δ_{i+1} - Δ_i| > 1` sample.
4. **Augment** with four configurations (Syn_1–Syn_4, see below).
5. **Second screening** — discard if any `Δ_i` exceeds `δ_max`; otherwise retain
   the sample for the final training set.

The figures focus on the selected depth point; the tables also report the
screening result over the complete 64-depth window.

---

## Augmentation configurations

| Configuration | Source (embedded) | Effect |
|---|---|---|
| **Syn_1** | `enhance_data_g1.py` | Piecewise attenuation + sinusoidal baseline drift + scattering noise (`noise_level=0.10`, 8 scatterers) |
| **Syn_2** | `enhance_data_g2.py` | Stronger scattering noise (`noise_level=0.25`, 16 scatterers) |
| **Syn_3** | `enhance_data_g3.py` | Piecewise attenuation + weaker scattering noise (`noise_level=0.10`, 8 scatterers) |
| **Syn_4** | `enhance_data_g4.py` | Stronger piecewise attenuation only |

The notebook calls the embedded production `process_single_sample` function for
each configuration, so the enabled/disabled operations, attenuation intervals,
noise regions, random ranges, and three-point parabolic label refinement remain
identical to the source scripts.

### Label refinement

After noise injection and waveform processing, the initial arrival label may
deviate slightly from the local waveform trough. The implementation rounds the
initial label to the nearest integer sample, `t_arr`, and uses that sample and
its two immediate neighbors to estimate the vertex of a parabola:

t̂_arr = t_arr + ( f[t_arr − 1] − f[t_arr + 1] ) / ( 2 · ( f[t_arr − 1] − 2·f[t_arr] + f[t_arr + 1] ) )

This provides a sub-sample correction without dense cubic interpolation. If the
center sample is at a waveform boundary, or if the denominator is degenerate or
non-finite, the integer label is retained.

---

## About the embedded production code

The notebook embeds the exact constants and core functions extracted from the
production scripts `dif_time_g1.py`, `diff_velocity_g1.py`, and
`enhance_data_g1.py`–`enhance_data_g4.py`. Each is loaded into its own module
namespace (via `types.ModuleType` + `exec`) so that the public notebook is
self-contained and the identical function/parameter names across scripts do not
collide. This is the mechanism that lets the released notebook run without the
unreleased `.py` files.

---

## Requirements

- Python 3.8+
- `numpy`, `scipy`, `PyWavelets`, `matplotlib`, `pandas`, `tqdm`, `psutil`

Install with `pip install -r requirements.txt`.

---

## License

_License to be added._

---

## Citation

_If this repository accompanies a manuscript, add the recommended citation here._
