from __future__ import annotations

import json
import argparse
from pathlib import Path

from cortex_m.assistant import RuleBasedCortex

ROOT = Path(__file__).resolve().parents[1]
lexicon = json.loads((ROOT / "data" / "lexicon.json").read_text(encoding="utf-8"))
assistant = RuleBasedCortex(lexicon["keys"], lexicon["values"])
parser = argparse.ArgumentParser(description="Run the model-free Cortex-M console demo.")
parser.add_argument("--message", action="append", default=[], help="User statement to process; may be repeated.")
parser.add_argument("--ask", action="append", default=[], help="Question to answer after processing messages; may be repeated.")
args = parser.parse_args()
try:
    if args.message or args.ask:
        for text in args.message:
            count = assistant.observe(text)
            print(f"Stored {count} explicit fact(s).")
        for query in args.ask:
            print(assistant.respond(query))
    else:
        print("Cortex-M model-free demo. Type text to record a fact; prefix questions with 'ask:'.")
        print("Facts outside the configured key/value vocabulary will not be stored.")
        while True:
            try:
                text = input("you> ").strip()
            except EOFError:
                break
            if text.casefold() in {"quit", "exit"}:
                break
            if text.casefold() == "clear":
                assistant.store.clear_all()
                print("cortex> Recorded facts cleared.")
            elif text.casefold().startswith("ask:"):
                print("cortex> " + assistant.respond(text[4:].strip()))
            else:
                count = assistant.observe(text)
                if count:
                    print(f"cortex> Stored {count} explicit fact(s).")
                else:
                    print("cortex> No supported explicit fact matched; rephrase or add this key/value to the configured lexicon.")
finally:
    assistant.close()
