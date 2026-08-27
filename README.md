# Quantum Data-Reuploading Classifier: Robustness & Error Mitigation

This project trains a small **variational quantum classifier** (a PyTorch
module backed by a parameterized Qiskit circuit) on the PennyLane
`plus-minus` image dataset, attacks it with an **adversarial (PGD) attack**,
retrains it to be more robust, and then evaluates how much of that accuracy
survives when the model is run under **realistic hardware noise** — with and
without several **quantum error-mitigation** techniques (ZNE, TREX, PEC,
CDR).

Two notebooks implement the same experiment under two different execution
regimes:

| Notebook | Estimator used for training/eval | What it measures |
|---|---|---|
| `Model_without_noise.ipynb` | Exact, noiseless statevector estimator | Best-case behavior: model accuracy and adversarial robustness with no hardware noise |
| `Model_with_noise.ipynb` | Noisy density-matrix simulator calibrated to IBM's `ibm_aachen` device | Realistic behavior: how much accuracy is lost to hardware noise, and how much of it error mitigation can recover |

Both notebooks share the same model architecture, training loop, PGD attack,
and error-mitigation estimator implementations; they differ mainly in which
Qiskit `Estimator` primitive is plugged into the model and in a few
noise-specific cells only present in the "with noise" notebook.

---

## 1. Project workflow / execution flow

Both notebooks follow the same high-level pipeline, cell by cell:

1. **Imports & reproducibility** (cell 0) — all imports plus global RNG
   seeding (`random`, NumPy, PyTorch CPU/GPU) for reproducible runs.
2. **Load & visualize data** — loads the PennyLane `plus-minus` dataset
   (16x16 grayscale images of four symbols: `-`, `+`, a stylized `-|`, and
   `/=`) and plots one example of each class.
3. **Define the model** — `QiskitReuploadingClassifier`, a PyTorch
   `nn.Module` wrapping a parameterized Qiskit circuit executed via
   `EstimatorQNN`/`TorchConnector`. In the noisy notebook, a noisy
   density-matrix `Estimator` calibrated from a snapshot of `ibm_aachen` is
   built first and passed into the model.
4. **Instantiate & sanity-check** the model with random dummy input to
   confirm the output shape is `[batch, num_classes]`.
5. **Set hyperparameters and train** — Adam optimizer, cross-entropy loss,
   a handful of epochs over a small subset (200 train / 50 test images) of
   the dataset, printing train/val loss & accuracy every epoch and
   visualizing predictions on four example images.
6. **Save/load a checkpoint** — model + optimizer state is checkpointed to
   disk so later cells can be re-run without retraining (some later cells
   are explicitly marked `#training skipped` / `#if training is skipped`).
7. **Evaluate clean accuracy**, then **craft a PGD adversarial attack**
   against the test set and re-measure accuracy on the perturbed inputs.
8. **Adversarial retraining** — a small adversarial-example augmented
   fine-tuning pass to (partially) recover robustness, followed by another
   checkpoint save/load and another PGD attack + evaluation.
9. **Error-mitigation experiments** (bulk of the remaining cells) — for
   each of several mitigation strategies, a custom Qiskit
   `BaseEstimatorV2`-compatible wrapper is defined, plugged into a fresh
   `QiskitReuploadingClassifier` (loaded with the trained weights), and
   used to re-evaluate accuracy on the adversarial test set:
   - **ZNE** (Zero-Noise Extrapolation) via a hand-rolled global circuit
     folder (`MitiqZNEEstimatorV2`), run first against the local noisy
     simulator, then against real IBM Runtime hardware.
   - **TREX** (Twirled Readout Error eXtinction) and a **no-mitigation**
     baseline, both via Qiskit Runtime's built-in resilience options.
   - **PEC** (Probabilistic Error Cancellation) via Mitiq
     (`MitiqPECEstimatorV2`, with-noise notebook only), sampling
     quasi-probability circuit representations of each noisy gate.
   - **CDR** (Clifford Data Regression) via a custom
     `MitiqCDREstimatorV2`, which fits a linear noisy-to-ideal correction
     from near-Clifford training circuits and applies it to the target
     circuit's expectation value. The with-noise and without-noise
     notebooks use two different (independently written) implementations
     of this class — see [Section 5](#5-notable-differences-between-the-two-notebooks).

Cells are meant to be run top-to-bottom in a single kernel session; several
cells reuse module-level globals (`qml_model`, `checkpoint`, `feats_test`,
`perturbed_feats_test`, etc.) set by earlier cells rather than being fully
self-contained (see [Limitations](#6-assumptions-limitations--known-issues)).

---

## 2. The model: `QiskitReuploadingClassifier`

Both notebooks define an (almost) identical PyTorch module:

- **Input**: a flattened 16x16 image (256 features).
- **Classical preprocessing**: a learnable linear layer
  (`input_projection`) maps the 256 input features onto
  `num_layers * num_qubits * 3` rotation angles (default: `2 * 8 * 3 = 48`
  angles for the default 8-qubit, 2-layer configuration).
- **Quantum circuit**: a "strongly entangling" ansatz (mirroring
  PennyLane's `StronglyEntanglingLayers` template) built from
  `RZ-RY-RZ` rotation blocks per qubit per layer, interleaved with rings of
  `CNOT` gates whose stride cycles with the layer index.
- **Readout**: one single-qubit Pauli-Z expectation value per output class
  (4 classes → 4 qubits read out), evaluated via Qiskit Machine Learning's
  `EstimatorQNN` and exposed to PyTorch autograd through `TorchConnector`.
- **Backend-aware construction**: if a `backend` is supplied, the logical
  circuit is transpiled to that backend's native gate set/connectivity
  (`optimization_level=0`, fixed `seed_transpiler=42` for reproducibility)
  and the readout observables are remapped onto the resulting physical
  qubit layout.

The gradient method differs between the two notebooks: the noiseless
notebook uses `ReverseEstimatorGradient` (fast, exact, requires a
statevector-capable estimator), while the noisy notebook uses
`ParamShiftEstimatorGradient` (the parameter-shift rule, which works with
any estimator, including noisy/hardware ones).

---

## 3. Adversarial attack: `PGD`

Both notebooks use the same L∞ **Projected Gradient Descent** attack
(`PGD` function): it iteratively takes signed-gradient ascent steps on the
model's cross-entropy loss with respect to the input, clamping the
cumulative perturbation to an L∞ ball of radius `epsilon` (default `0.05`).
This is used both to evaluate adversarial robustness and to generate
adversarial training examples for the retraining step.

---

## 4. Error-mitigation estimators

All of the mitigation classes implement Qiskit's `BaseEstimatorV2.run(...)`
interface (or a close approximation of it), so they can be dropped directly
into `QiskitReuploadingClassifier(estimator=..., backend=...)` without
touching the model code.

| Class | Technique | How it works |
|---|---|---|
| `MitiqZNEEstimatorV2` | Zero-Noise Extrapolation | Globally folds each circuit to several odd noise-scale factors (`U`, `U U†U`, `U U†U U†U`, ...), executes all of them, and Lagrange-extrapolates the resulting expectation values back to the zero-noise limit. |
| `MitiqPECEstimatorV2` (with-noise notebook only) | Probabilistic Error Cancellation | Builds a Mitiq quasi-probability representation of each noisy gate from measured gate error rates, samples many circuits from that representation, executes them, and recombines the results to cancel local depolarizing noise in expectation. |
| `MitiqCDREstimatorV2` | Clifford Data Regression | Generates several "training" circuits by replacing a fraction of a circuit's gates with the nearest Clifford equivalent (which can be evaluated cheaply/exactly), evaluates them on both a noisy and an ideal estimator, fits a linear noisy→ideal correction, and applies it to the real (non-Clifford) target circuit. |
| `_SynchronousJob`, `_CDRData`, `_CDRResult` | — | Minimal stand-ins for Qiskit Runtime's `Job`/`PubResult`/`DataBin` objects, so the custom estimators can present the same synchronous `.run(...).result()` interface as a real Qiskit primitive. |

TREX (readout-error mitigation) and the no-mitigation baseline don't need a
custom class — they're configured directly via Qiskit Runtime's
`EstimatorOptions(resilience={...})`.

---

## 5. Notable differences between the two notebooks

- **`MitiqPECEstimatorV2`** (Probabilistic Error Cancellation) only appears
  in `Model_with_noise.ipynb`
- Saved checkpoint filenames differ (`..._with_noise_psr.pth` vs. no
  suffix) so the two notebooks' artifacts don't collide if run in the same
  working directory.

---

## 6. Assumptions, limitations, & known issues

- **Requires an IBM Quantum Runtime account.** Several cells call
  `QiskitRuntimeService()` and target the `ibm_aachen` backend, both to
  build the noisy simulator's noise model and to submit jobs to *real*
  hardware (`USE_REAL_HARDWARE = True` path, Runtime `Session`/`Batch`
  blocks). Without valid credentials configured
  (`QiskitRuntimeService.save_account(...)`), these cells will fail. By
  default `USE_REAL_HARDWARE` is `False`, so most of the notebook runs
  against a *local* noisy simulator instead.
- **Small, fixed data subsets.** Training uses only 200 images and
  evaluation only 50 (further sub-sampled to 50 or fewer in several
  mitigation cells), which keeps circuit-execution costs manageable but
  means accuracy numbers have high variance and should not be read as
  precise benchmarks.
- **Long-running / expensive cells.** PEC (100 circuit samples per
  expectation value) and CDR (multiple training circuits per prediction,
  each requiring its own estimator call) multiply the number of circuit
  executions substantially; on real hardware these cells can be slow and
  consume significant QPU time/shots.
- **Non-deterministic real-hardware results.** Cells that submit to actual
  IBM hardware will not reproduce bit-for-bit even with fixed seeds, since
  queueing, calibration drift, and hardware noise vary over time. The
  stored notebook outputs reflect one particular run and are provided for
  reference only.

---

## 7. Dependencies & installation

Both notebooks require a Python environment with the following packages
(the exact versions used to originally run these notebooks are not pinned
in the notebook itself; install recent, mutually compatible versions):

```bash
pip install \
    numpy \
    matplotlib \
    torch \
    pennylane \
    qiskit \
    qiskit-aer \
    qiskit-algorithms \
    qiskit-machine-learning \
    qiskit-ibm-runtime \
    mitiq \
    cirq   # only needed by Model_with_noise.ipynb (Mitiq PEC / Cirq circuit handling)
```

You will also need:

- **IBM Quantum Runtime credentials** configured for `QiskitRuntimeService`
  (see the [Qiskit IBM Runtime documentation](https://docs.quantum.ibm.com/)
  for `QiskitRuntimeService.save_account(...)`) if you want to run the
  cells that build a noise model from `ibm_aachen` or submit jobs to real
  hardware. Cells that only use the local `AerSimulator` do not require
  this, but the noisy notebook's noise model is *seeded from* a live
  backend snapshot, so even purely local runs need read access to the
  backend's calibration data the first time.
- A Jupyter environment (Jupyter Notebook, JupyterLab, or VS Code's
  notebook support) matching the kernel referenced in the notebook
  metadata (`Python (My QML Venv)`), or any Python 3 kernel with the
  packages above installed.

---

## 8. Running the notebooks

1. Install the dependencies above and configure IBM Quantum Runtime
   credentials if you intend to run hardware-facing cells.
2. Open either notebook and **run all cells top to bottom** in a single
   kernel session (`Run All` / `Restart & Run All`). Do not skip cells —
   later cells depend on variables and imports defined earlier.
3. To skip retraining on a subsequent run, you can start execution from a
   checkpoint-loading cell (marked `#training skipped` /
   `#if training is skipped`) once a `.pth` checkpoint file already exists
   in the working directory from a prior run.
4. Set `USE_REAL_HARDWARE = True` in the CDR real-hardware cell only if you
   have IBM Quantum Runtime access and are prepared for the associated
   queue time / QPU usage; otherwise leave it `False` to stay on the local
   noisy simulator.
5. Expect the PEC and CDR mitigation cells to take noticeably longer than
   the others, since they multiply the number of circuit evaluations per
   prediction.

---

## 9. Outputs produced

Running a notebook end-to-end writes the following files to the working
directory:

- `qiskit_reuploading_classifier[...].pth` — model + optimizer checkpoint
  saved immediately after initial training.
- `qiskit_reuploading_classifier_after_attack[...].pth` — checkpoint saved
  after adversarial retraining.
- `pgd_attack_data.pt` (with-noise notebook only) — cached adversarial test
  features/labels/perturbations, so the PGD attack doesn't need to be
  regenerated for later mitigation-comparison cells.

Console output throughout reports training/validation loss and accuracy
per epoch, clean vs. adversarial accuracy, and the mitigated accuracy
achieved by each error-mitigation strategy, letting you directly compare
how much adversarial/noise-induced accuracy loss each technique recovers.
