#!/bin/zsh
S=/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/demo_combined
export OURBOT_WORK=$S/work ENGINE_DIR=/Users/biofanat/Documents/classes/f26/cs535/final/engine
for s in "$@"; do ~/sandbox/bot-venv/bin/python $S/headless.py --seed $s --tag demo --a agentform --b stock > $S/work/seed$s.out 2>$S/work/seed$s.err; done
