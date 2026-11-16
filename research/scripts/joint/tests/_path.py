"""sys.path setup shared by the tests: engine/ and engine/showdown/ (behavior, hidden_agents, nash)."""
import os
import sys

ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "engine"))
sys.path[:0] = [ENGINE, os.path.join(ENGINE, "showdown")]
