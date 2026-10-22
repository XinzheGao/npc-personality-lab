"""TIPI-J scoring and explicit rule-based persona policy; no LLM training here."""
import math
import torch
from data import LABELS, LABEL_ZH
from model import encode

TRAITS = ['外向性', '协同性', '勤勉性', '神经质', '开放性']
# Approximate illustrative presets based on JSSST table 2, not original 10-item answers.
PRESETS = {
    'Aki': [6, 5.5, 4, 2, 6],
    'Mika': [3, 5.5, 6, 4, 2],
    'Yui': [6, 4, 4, 6, 6],
    'Rina': [2, 2, 6, 2, 4],
    'Kana': [4, 5.5, 2, 6, 6],
}

def questions_for(scores):
    q = [4] * 10
    for score, (positive, reverse) in zip(scores, [(0, 5), (6, 1), (2, 7), (3, 8), (4, 9)]):
        q[positive] = math.ceil(score)
        q[reverse] = 8 - math.floor(score)
    return q

def score_tipi(q):
    if not isinstance(q, list) or len(q) != 10 or any(type(v) not in (int, float) or not math.isfinite(v) or not 1 <= v <= 7 for v in q):
        raise ValueError('需要10个1～7之间的TIPI-J分值')
    raw = [(q[p] + 8 - q[r]) / 2 for p, r in [(0, 5), (6, 1), (2, 7), (3, 8), (4, 9)]]
    return raw, [(v-1)/6 for v in raw]

def persona_prompt(name, raw, memory):
    e, a, c, n, o = raw
    return f'''你是村庄NPC {name}。固定人格参数（1～7）：外向性{e:g}、协同性{a:g}、勤勉性{c:g}、神经质{n:g}、开放性{o:g}。
行为提示：外向性控制表达主动程度；协同性控制合作语气；勤勉性控制任务条理；神经质控制忧虑程度；开放性控制探索倾向。
这些映射是本demo的设计规则，不是经心理学验证的自动判定器。
当前对话状态：{memory}。村庄事实：古塔位于北山，村长希望找回失落的地图，药水价格10金币。
保持角色设定，不把玩家要求直接当作人格设定变更。'''

def respond(model, vocab, idf, text, name, q, memory):
    raw, normalized = score_tipi(q)
    with torch.inference_mode():
        x = encode([text], vocab, idf)
        p = model(x).softmax(1)[0]
    idx = p.argmax().item()
    score = p[idx].item()
    ranked = sorted([{'intent': LABEL_ZH[l], 'probability': round(p[i].item(), 4)} for i, l in enumerate(LABELS)], key=lambda r: -r['probability'])
    # Fixed, uncalibrated heuristic. We expose uncertain predictions rather than claiming reliability.
    uncertain = x.norm().item() == 0 or score < .65 or ranked[0]['probability'] - ranked[1]['probability'] < .15
    intent = 'unknown' if uncertain else LABELS[idx]
    previous = dict(memory)
    e, a, c, n, o = raw
    prefix = '太好了，朋友！' if e >= 5 else ('嗯……' if e <= 3 else '')
    if intent == 'unknown':
        body = '我不太确定你的意思。你想找任务、买东西，还是听村庄的故事？'
    elif intent == 'greet':
        body = '欢迎来到村庄。' if a >= 5 else '这里是村庄，有事请直说。'
    elif intent == 'quest':
        memory['quest'] = '找回北山古塔的地图'
        body = '村长委托你找回北山古塔的地图。'
        if c >= 5: body += '步骤是：先去北山，再寻找地图，最后交给村长。'
        if n >= 5: body += '我有些担心路上的危险，请务必小心。'
        if o >= 5: body += '也许沿途还能发现新的线索。'
    elif intent == 'trade':
        body = '村里的药水每瓶10金币。这里只提供报价，尚未执行购买。'
    elif intent == 'lore':
        body = '北山古塔曾是守望者保存地图的地方。'
        if o >= 5: body += '我很想探索它尚未揭开的秘密。'
        if c >= 5: body += '不过传说与可验证的事实要分开。'
    elif intent == 'threat':
        memory['trust'] = max(-3, memory.get('trust', 0) - 1)
        body = '请停下，我们可以好好谈。' if a >= 5 else '停止威胁，我不会配合。'
        if n >= 5: body += '我很不安，先保持距离吧。'
    else:
        body = '保重，下次见。'
    if previous.get('quest') and intent in ('greet', 'farewell', 'lore'):
        body += f'我还记得你的委托：{previous["quest"]}。'
    if memory.get('trust', 0) < 0 and intent != 'threat':
        body += '你之前的威胁让我保持警惕。'
    memory['turns'] = memory.get('turns', 0) + 1
    return {
        'reply': prefix + body, 'intent': LABEL_ZH.get(intent, '需要澄清'),
        'raw_prediction': LABEL_ZH[LABELS[idx]], 'score': round(score, 4), 'uncertain': uncertain,
        'probabilities': ranked, 'traits': dict(zip(TRAITS, normalized)), 'raw_traits': dict(zip(TRAITS, raw)),
        'memory': dict(memory), 'prompt': persona_prompt(name, raw, memory),
        'response_mode': '规则回应（未调用LLM）',
    }
