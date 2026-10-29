# CORTEX-M System & Environment Specifications

**Document Version:** 1.0  
**Capture Timestamp:** 2026-10-29T02:49:00Z  
**Governing Standard:** CORTEX-M Master Execution Guide §5-6, Phase 0 & Phase 1  
**Integrity Mode:** Development / Research Substrate  

---

## 1. Host Operating System
- **Operating System:** Microsoft Windows 11 Home Single Language
- **OS Version:** 10.0.26300
- **OS Build Number:** 26300
- **Architecture:** 64-bit Operating System, x64-based processor (AMD64)
- **Host Platform:** Windows-11-10.0.26300-SP0

---

## 2. Hardware Architecture & CPU Topology
- **Processor:** 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz
- **Physical Cores:** 4
- **Logical Processors:** 8 (Hyper-Threading enabled)
- **Base Clock Frequency:** 2419 MHz (~2.42 GHz)
- **Max Turbo Clock:** ~4.20 GHz
- **Instruction Set Extensions:** AVX-512, AVX2, FMA3, SSE4.2
- **Accelerator / GPU:** Host contains NVIDIA GeForce MX350 (2048 MiB / 2 GB VRAM, Driver 582.67, CUDA 13.0 capable) and Integrated Intel Iris Xe Graphics. Active Python virtual environment (.venv-mimo) currently runs `torch 2.13.0+cpu` (CPU build). Compute is locked to CPU execution mode within a ~900 MB free RAM buffer.
- **Compute Policy:** Strict CPU-Only execution as mandated by Master Execution Guide §14 & §18 and user hardware directive. All neural routing modules, Hopfield associative memories, and symbolic search routines are designed to execute within CPU and memory bounds (~900 MB free buffer).

---

## 3. Physical & Virtual Memory Subsystem
- **Total Physical RAM:** 8,117,028 KB (~7.74 GB / 8.00 GB installed)
- **Available Physical RAM:** ~1.00 GB – 2.50 GB (dynamic depending on system processes)
- **Memory Management Policy:**
  - Neural batch sizes must be constrained ($B \le 32$, default $B=16$ or $B=8$).
  - Hopfield associative memory dimension $D \le 128$ (default $D=64$, latent gist $D_{gist}=32$).
  - Avoid unnecessary in-memory caching of full dataset corpora; stream from disk or use chunked generators.

---

## 4. Storage & Filesystem Layout
- **Drive E: (Primary Research & Repository Workspace):**
  - Total Capacity: 293.00 GB (314,572,795,904 bytes)
  - Used Space: 35.20 GB (35,200,811,008 bytes)
  - Free Space: 279.37 GB (279,371,984,896 bytes)
  - Status: Healthy, sufficient capacity for all synthetic datasets, benchmark logs, and model checkpoints.
- **Drive C: (System & OS Partition):**
  - Total Capacity: 237.33 GB (254,833,381,376 bytes)
  - Used Space: 202.51 GB (202,508,148,736 bytes)
  - Free Space: 52.33 GB (52,325,232,640 bytes)
- **Drive D: (Secondary Storage):** 294.93 GB Free / 314.57 GB Total
- **Drive F: (Tertiary Storage):** 362.16 GB Free / 371.06 GB Total

---

## 5. Python Runtime Environment
- **Python Version:** 3.12.10 (CPython, tags/v3.12.10:0cc8128, Apr 8 2025, 12:21:36)
- **Compiler:** MSC v.1943 64 bit (AMD64)
- **Python Executable:** `E:\Research\.venv-mimo\Scripts\python.exe`
- **Virtual Environment Base:** `E:\Research\.venv-mimo`
- **Package Installer:** `pip 25.0.1`

---

## 6. PyTorch Subsystem
- **PyTorch Version:** `2.13.0+cpu`
- **CUDA Availability:** `False` (`torch.cuda.is_available() == False`)
- **ROCm / MPS Availability:** `False`
- **CPU Computation Threads:** 4 (`torch.get_num_threads() == 4`)
- **Intel MKL Acceleration Backend:** `True` (`torch.backends.mkl.is_available() == True`)
- **OpenMP Parallelism Backend:** `True` (`torch.backends.openmp.is_available() == True`)
- **Deterministic Algorithms Support:** Available via `torch.use_deterministic_algorithms(True)`

---

## 7. Complete Locked Package Manifest (`pip freeze`)
```text
charset-normalizer==3.5.2
colorama==0.4.6
contourpy==1.4.0
-e git+https://github.com/Dhaval90750/cortex-m.git@70022f7c16183f46e4f57d9e9b4e6598dd7d6a06#egg=cortex_m
cycler==0.12.1
filelock==3.32.3
fonttools==4.67.0
fsspec==2026.7.0
iniconfig==2.3.1
Jinja2==3.1.6
kiwisolver==1.5.1
lxml==6.1.3
markdown-it-py==4.2.0
MarkupSafe==3.0.3
matplotlib==3.11.2
mdurl==0.1.2
mpmath==1.3.0
networkx==3.6.1
numpy==2.5.2
packaging==26.3
pandas==3.0.5
pillow==13.0.0
pluggy==1.7.0
Pygments==2.21.0
pyparsing==3.3.3
pytest==9.2.0
python-dateutil==3.0.0
python-docx==1.2.0
reportlab==5.1.0
rich==15.0.0
setuptools==78.1.0
six==1.17.0
sympy==1.14.0
torch==2.13.0+cpu
typing_extensions==4.16.0
tzdata==2026.5
```

---

## 8. Compute Constraints & Execution Directives
1. **CPU Workstation Discipline:** Models and benchmarks must be engineered for CPU execution. Parameter counts for prototype modules must remain under $10\text{M}$ (typical CORTEX-M routing layers $\approx 25\text{K} - 100\text{K}$ parameters).
2. **Explicit Invocation:** All scripts and test runners must strictly use the virtual environment executable:
   ```powershell
   & "e:\Research\.venv-mimo\Scripts\python.exe" -m pytest
   ```
3. **Reproducibility Seeds:** Multi-seed evaluations must use fixed seed lists (e.g., seeds `[42, 101, 202, 303, 404, 505, 606, 707, 808, 909]`) with deterministic torch and numpy states.
