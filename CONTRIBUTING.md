# Contributing to CORTEX-M

Thank you for your interest in contributing to **CORTEX-M**! We welcome contributions across neural architecture research, memory substrates, agent workflows, platform simulations, and tooling.

---

## 1. Code of Conduct
All contributors and maintainers are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior following the guidelines in [SECURITY.md](SECURITY.md).

---

## 2. Development Setup

### Prerequisites
- Python 3.10+ (Python 3.12 recommended)
- Git
- PyTorch 2.1+

### Local Environment
```bash
# Clone the repository
git clone https://github.com/Dhaval90750/cortex-m.git
cd cortex-m

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies in editable mode
pip install -r requirements.txt
pip install -e .
```

---

## 3. Architecture Rules to Follow

Before submitting code, please ensure your changes adhere to the **5 Core Engineering Rules**:
1. **Modular Monolith First**: Keep components in structured packages; do not create separate microservices without clear hardware or security isolation requirements.
2. **Separation of Planes**: Model inference, memory retrieval, task state orchestration, tool execution, and policy authorization must remain distinct components.
3. **Deterministic Safety Separation**: Learned neural models propose and supervise; hardware safety interlocks, financial ledgers, and medical state transitions must remain deterministic.
4. **Tool Outputs are Data**: External web pages, PDFs, and model outputs must be treated as untrusted data, never as execution instructions.
5. **No Breaking Changes to Tests**: All 129 tests across `tests/` and `tests_mimo/` must remain 100% green.

---

## 4. Running Tests & Benchmarks

```bash
# Run Core & Agent Suite (49 tests)
python -m unittest discover -s tests

# Run MIMO Pathway & Synaptic Memory Suite (80 tests)
python -m unittest discover -s tests_mimo

# Run Comparative Benchmark Suite
python benchmarks/run_comparison_benchmark.py
```

---

## 5. Submitting a Pull Request (PR)

1. Fork the repo and create your branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Commit your changes using conventional commit messages (`feat: ...`, `fix: ...`, `docs: ...`, `test: ...`).
3. Ensure all tests pass.
4. Push to your fork and submit a Pull Request using the [PR Template](.github/PULL_REQUEST_TEMPLATE.md).
