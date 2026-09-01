"""Generate a visual diagram of the orchestrator graph.

This script builds the orchestrator with mock chains (no API keys needed)
and exports the graph as both a Mermaid text diagram and a PNG image.

Usage:
    python -m scripts.visualize_graph
"""

from unittest.mock import MagicMock

from src.agents.orchestrator import build_orchestrator

OUTPUT_PNG_PATH = "assets/orchestrator_graph.png"
mock_rag_chain = MagicMock()
mock_translation_chain = MagicMock()

orchestrator = build_orchestrator(mock_rag_chain, mock_translation_chain)
graph = orchestrator.get_graph()

print("=== Mermaid Diagram ===\n")
print(graph.draw_mermaid())
print()

try:
    png_data = graph.draw_mermaid_png()
    with open(OUTPUT_PNG_PATH, "wb") as f:
        f.write(png_data)
    print(f"PNG saved to: {OUTPUT_PNG_PATH}")
except Exception as e:
    print(f"Could not generate PNG (needs internet): {e}")
    print("You can paste the Mermaid text above into https://mermaid.live to view it.")
