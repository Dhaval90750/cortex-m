"""Real Data Ingestion & Normal User Episodic Memory Trainer for CortexLM.

Ingests real multi-turn episodic datasets (e.g., data/generated/dev.jsonl, data/lexicon.json)
and normal everyday user knowledge directly into CortexLM's continuous synaptic weights.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cortex_m.model.cortex_lm import CortexLM


# Everyday Knowledge for Normal Users (Lifestyle, Health, Travel, Finance, Routines)
EVERYDAY_USER_KNOWLEDGE: List[Dict[str, str]] = [
    {
        "memory_id": "budgeting_rule_50_30_20",
        "text": "The 50/30/20 budgeting rule allocates 50% of after-tax income to essential needs (rent, groceries, utilities), 30% to personal wants (dining out, entertainment, hobbies), and 20% to savings, investments, or debt payoff.",
        "source": "personal_finance_guide",
    },
    {
        "memory_id": "sleep_hygiene_circadian",
        "text": "Good sleep hygiene optimizes circadian rhythms by maintaining a consistent bedtime, avoiding blue light from screens 1 hour before sleep, keeping the bedroom cool (around 65°F/18°C), and getting morning sunlight exposure.",
        "source": "wellness_and_health_habits",
    },
    {
        "memory_id": "hydration_guidelines",
        "text": "Daily healthy hydration guidelines recommend drinking approximately 2 to 3 liters of water per day for adults, adjusted for physical activity levels, climate, and overall health.",
        "source": "everyday_nutrition_facts",
    },
    {
        "memory_id": "travel_packing_essentials",
        "text": "Smart travel packing principles recommend rolling clothes to save space, keeping travel documents and medications in a carry-on bag, and using packing cubes to separate categories.",
        "source": "travel_planner_tips",
    },
    {
        "memory_id": "pomodoro_technique",
        "text": "The Pomodoro Technique boosts everyday productivity by breaking work into focused 25-minute intervals separated by short 5-minute breaks, with a longer 20-minute break after every four sessions.",
        "source": "productivity_methods",
    },
    {
        "memory_id": "daily_walking_exercise",
        "text": "Walking 7,000 to 10,000 steps daily significantly improves cardiovascular health, enhances joint mobility, boosts daily energy, and lowers stress.",
        "source": "wellness_and_health_habits",
    },
    {
        "memory_id": "emergency_fund_rule",
        "text": "A personal emergency fund should safely cover 3 to 6 months of essential living expenses, kept in a secure, liquid high-yield savings account.",
        "source": "personal_finance_guide",
    },
    {
        "memory_id": "digital_declutter_habit",
        "text": "Regular digital decluttering by archiving old inbox emails, organizing desktop files into folders, and closing unused browser tabs reduces cognitive overload.",
        "source": "productivity_methods",
    },
]


class RealDataTrainer:
    """Trainer that loads real-world episodic datasets into continuous synaptic weights."""

    def __init__(self, lm: CortexLM):
        self.lm = lm

    def ingest_jsonl_dataset(
        self,
        filepath: str,
        max_samples: int = 50,
        source_label: str = "user_episodic_data",
    ) -> Dict[str, Any]:
        """Ingest real JSONL conversational/episodic records into physical synapses."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        learned_count = 0
        imprinted_keys: List[str] = []

        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Extract evidence / fact text
                    annotations = data.get("annotations", [])
                    if annotations:
                        ann = annotations[0]
                        evidence_text = ann.get("evidence_text", "")
                        key = ann.get("key", f"pref_{idx}")
                        val = ann.get("value", "")
                        fact_id = ann.get("fact_id", f"fact_{idx}")

                        # Create crisp semantic assertion
                        clean_assertion = f"User preference for '{key}' is '{val}'. {evidence_text}"
                        mem_id = f"realdata_{key.replace(' ', '_')}"

                        self.lm.learn(
                            clean_assertion,
                            memory_id=mem_id,
                            source=source_label,
                        )
                        learned_count += 1
                        imprinted_keys.append(key)
                    elif "turns" in data:
                        turns = data.get("turns", [])
                        text = " ".join([t.get("text", "") for t in turns])
                        if text:
                            mem_id = f"realdata_turn_{idx}"
                            self.lm.learn(text, memory_id=mem_id, source=source_label)
                            learned_count += 1
                except Exception:
                    continue

        return {
            "status": "real_data_imprinted",
            "file": filepath,
            "total_imprinted": learned_count,
            "keys": list(set(imprinted_keys)),
        }

    def imprint_everyday_knowledge(self) -> int:
        """Imprint everyday life, health, finance, and productivity facts."""
        for item in EVERYDAY_USER_KNOWLEDGE:
            self.lm.learn(
                item["text"],
                memory_id=item["memory_id"],
                source=item["source"],
            )
        return len(EVERYDAY_USER_KNOWLEDGE)
