"""Focused checks for scoring, loaded inference, and session-state behavior."""
from pathlib import Path
from model import load_model
from npc import PRESETS, questions_for, respond, score_tipi

root = Path(__file__).resolve().parent
model, vocab, idf = load_model(root / 'artifacts/intent.pt')
assert score_tipi([4] * 10) == ([4] * 5, [.5] * 5)
assert score_tipi([7, 1, 7, 1, 7, 1, 7, 1, 7, 1])[0] == [7] * 5
for invalid in ([], [0] * 10, [8] * 10, ['4'] * 10):
    try:
        score_tipi(invalid)
        raise AssertionError('invalid TIPI values were accepted')
    except ValueError:
        pass
memory = {}
aki = respond(model, vocab, idf, '有什么任务可以接', 'Aki', questions_for(PRESETS['Aki']), memory)
bye = respond(model, vocab, idf, '再见', 'Aki', questions_for(PRESETS['Aki']), memory)
assert aki['intent'] == '寻求任务'
assert '我还记得' in bye['reply'] and memory['turns'] == 2
rina = respond(model, vocab, idf, '有什么任务可以接', 'Rina', questions_for(PRESETS['Rina']), {})
assert aki['reply'] != rina['reply'] and '步骤' in rina['reply']
fresh = respond(model, vocab, idf, '再见', 'Aki', questions_for(PRESETS['Aki']), {})
assert '我还记得' not in fresh['reply']
unknown = respond(model, vocab, idf, 'zxqvzxqv', 'Aki', questions_for(PRESETS['Aki']), {})
assert unknown['uncertain'] and unknown['intent'] == '需要澄清'
print('Passed: TIPI scoring and validation, model load, task memory, persona difference, state reset, unknown input')
