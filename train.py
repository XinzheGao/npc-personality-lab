"""Validation selects checkpoint; test is evaluated once after selection."""
import copy
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from data import TRAIN, VAL, TEST, OOD, LABELS, LABEL_ZH, rows
from model import IntentNet, encode, fit_vocab

ROOT = Path(__file__).resolve().parent

def train():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    splits = {name: rows(part) for name, part in [('train', TRAIN), ('validation', VAL), ('test', TEST)]}
    sets = {name: {t for t, _ in part} for name, part in splits.items()}
    assert not (sets['train'] & sets['validation'] or sets['train'] & sets['test'] or sets['validation'] & sets['test'])
    vocab, idf = fit_vocab([t for t, _ in splits['train']])
    tensors = {name: (encode([t for t, _ in part], vocab, idf), torch.tensor([y for _, y in part])) for name, part in splits.items()}
    model = IntentNet(len(vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.008, weight_decay=.01)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(*tensors['train']), batch_size=18, shuffle=True)
    history, best, checkpoint, best_epoch = [], float('inf'), None, 0
    for epoch in range(1, 81):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            results = {}
            for name in ('train', 'validation'):
                x, y = tensors[name]
                logits = model(x)
                results[name + '_loss'] = round(criterion(logits, y).item(), 6)
                results[name + '_accuracy'] = round((logits.argmax(1) == y).float().mean().item(), 6)
        history.append({'epoch': epoch, **results})
        if results['validation_loss'] < best:
            best = results['validation_loss']
            checkpoint = copy.deepcopy(model.state_dict())
            best_epoch = epoch
    model.load_state_dict(checkpoint)
    model.eval()
    with torch.inference_mode():
        probs = model(tensors['test'][0]).softmax(1)
        predictions = probs.argmax(1)
        confusion = [[0] * len(LABELS) for _ in LABELS]
        details = []
        for i, (text, target) in enumerate(splits['test']):
            pred = predictions[i].item()
            confusion[target][pred] += 1
            details.append({'text': text, 'expected': LABEL_ZH[LABELS[target]], 'predicted': LABEL_ZH[LABELS[pred]], 'score': round(probs[i, pred].item(), 4), 'correct': target == pred})
        precision_recall = []
        for i, label in enumerate(LABELS):
            tp = confusion[i][i]
            precision = tp / max(1, sum(row[i] for row in confusion))
            recall = tp / max(1, sum(confusion[i]))
            f1 = 2 * precision * recall / max(1e-9, precision + recall)
            precision_recall.append({'label': LABEL_ZH[label], 'precision': precision, 'recall': recall, 'f1': f1})
        ood_probs = model(encode(OOD, vocab, idf)).softmax(1)
        ood = [{'text': t, 'predicted': LABEL_ZH[LABELS[p.argmax().item()]], 'score': round(p.max().item(), 4)} for t, p in zip(OOD, ood_probs)]
    report = {
        'seed': 42, 'torch_version': str(torch.__version__), 'device': 'cpu', 'best_epoch': best_epoch,
        'data_source': '小规模自建演示语料；不是论文原始数据',
        'split_policy': '预先固定的不同句子；词表与IDF仅来自训练集，验证损失选模型，测试集不参与调参',
        'limitations': '极小、同领域、自建语料，仅演示训练流程。相近词汇跨集合存在，不能证明真实游戏泛化；softmax分数未经校准；人格回应由规则控制。',
        'counts': {n: len(p) for n, p in splits.items()}, 'vocab_size': len(vocab),
        'test_accuracy': sum(d['correct'] for d in details) / len(details),
        'macro_f1': sum(r['f1'] for r in precision_recall) / len(precision_recall),
        'labels': [LABEL_ZH[l] for l in LABELS], 'confusion_matrix': confusion,
        'per_class': precision_recall, 'test_cases': details, 'ood_probes': ood, 'history': history,
        'data_sha256': hashlib.sha256((ROOT / 'data.py').read_bytes()).hexdigest(),
    }
    out = ROOT / 'artifacts'
    out.mkdir(exist_ok=True)
    torch.save({'state_dict': checkpoint, 'vocab': vocab, 'idf': idf}, out / 'intent.pt')
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in ['counts', 'best_epoch', 'test_accuracy', 'macro_f1']}, ensure_ascii=False))

if __name__ == '__main__':
    train()
