CORTEX-M MASTER RESEARCH PACK
Version: 1.0
Prepared: 29 October 2026

CONTENTS
1. CORTEX_M_Master_Execution_Guide.docx
   The detailed implementation and research guide for CORTEX-M Core, CORTEX-M Engineering and CORTEX-M Scientific Discovery (CORTEX-M-SD). It includes the research charter, architecture, data governance, phases, step-by-step procedures, expected artifacts, success/failure gates, experiment series, metrics, risk responses, timeline and release checks.

2. CORTEX_M_Master_Execution_Guide.md
   Plain-text Markdown source of the guide for Git version control, issue tracking and future edits.

3. CORTEX_M_Data_Registry.csv
   Operational registry of 104 candidate datasets, repositories, tools and evaluation resources. Treat the links and priorities as an acquisition plan, not authorization to download or train. Before ingestion, record the current release/version, license terms, intended use, data split, provenance, size and contamination checks.

4. CORTEX_M_Experiment_Preregistration_Template.json
   Machine-readable starting template for recording hypotheses, primary/secondary metrics, baselines, ablations, data IDs/versions, seeds, resource budgets, acceptance gates, limitations and actual results.

IMPORTANT RESEARCH NOTES
- This is an execution plan, not evidence that CORTEX-M has already demonstrated the proposed capabilities.
- Preserve simpler baselines and negative results. Do not claim component-specific improvement without ablations.
- Keep sealed evaluation data out of training, prompt development and threshold tuning.
- The initial compute assumption is CPU-only. Begin with small trainable modules and synthetic worlds; do not attempt large-scale general-purpose LLM pretraining on an ordinary CPU workstation.
- For CORTEX-M-SD, the simulator must generate ground truth separately from the model-visible record. Discovery narratives alone are not sufficient evidence of rule discovery.
- Dataset licences, terms, versions, API rules, and data availability can change; re-check current conditions before acquisition or use.
