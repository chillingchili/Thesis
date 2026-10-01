"""Training-only v4 feature and probability contracts; v3 artifacts stay frozen."""
import hashlib
import json
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score
from serve_study import ROOT, DATA, SEED, sha, write_json, metrics

DEST = ROOT / 'models/serve_refinement_v4'
OUTPUT = ROOT / 'outputs/serve_refinement_v4'
JOINTS = [11, 12, 14, 16, 20, 23, 24]
CONDITIONS = ['angles_thesis_aug', 'lean_control', 'lean_thesis_aug',
              'skeleton_thesis_aug', 'skeleton_motion_thesis_aug']
PROTOCOL = dict(version='serve_refinement_v4', seed=SEED, epochs=100,
    conditions=CONDITIONS, lean_joints=JOINTS,
    lean_features='10 angles + 14 hip-centered aspect-corrected xy + 7 masks + 14 adjacent displacements',
    model='GRU32, dropout .15, L2 .0001, Adam .001, batch16; v3 matched sampler',
    augmentation='reuse all 418 frozen v3 pixel/video variants; split original IDs first',
    outer='leave one of three training players out',
    inner='leave one of the remaining two players out; base learner sees one player',
    weight_grid=[0., .25, .5, .75, 1.],
    weight_selection='maximum mean-player macro-F1; ties minimum mean-player log loss, then closest to .5',
    calibration='one temperature on log of fused probabilities; minimize player-balanced inner OOF log loss, T in [.25,10]',
    confidence_selection='max coverage at >=80% observed inner accuracy, >=20 cases, >=5 per inner player; otherwise disabled',
    confidence_grid=[.4, .5, .6, .7, .8, .9, .95],
    candidate_selection='maximum outer mean-player macro-F1 of tuned calibrated hybrid; ties mean-player accuracy, condition order',
    final='one full-training GRU per condition; deployment calibration fits participant OOF; distribution shift is reported',
    diagnostic='previously inspected 175 clips, never used for fitting or selection',
    limitations='only three players; inner base models train on one player; candidate selection can inflate winning outer score')


def freeze():
    value = dict(PROTOCOL, v3_protocol_sha256=sha(DATA/'protocol.json'),
                 train_sha256=sha(ROOT/'data/serve_v2/train.npz'), source_sha256=sha(__file__))
    target = DEST/'protocol.json'
    if target.exists() and json.loads(target.read_text()) != value:
        raise ValueError('Frozen v4 protocol changed')
    write_json(target, value)
    return value


def lean(angles, skeleton, motion):
    xy = skeleton[..., :66].reshape(*skeleton.shape[:-1], 33, 2)[..., JOINTS, :]
    mask = skeleton[..., 66:][..., JOINTS]
    delta = motion[..., 99:].reshape(*motion.shape[:-1], 33, 2)[..., JOINTS, :]
    return np.concatenate([angles, xy.reshape(*angles.shape[:-1], 14), mask,
                           delta.reshape(*angles.shape[:-1], 14)], axis=-1).astype(np.float32)


def features(condition, angles, skeleton, motion):
    if condition == 'angles_thesis_aug': return angles
    if condition.startswith('lean'): return lean(angles, skeleton, motion)
    if condition == 'skeleton_thesis_aug': return skeleton
    if condition == 'skeleton_motion_thesis_aug': return motion
    raise ValueError(condition)


def temperature(p, t):
    logits = np.log(np.clip(np.asarray(p, float), 1e-7, 1.)) / t
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp/exp.sum(axis=-1, keepdims=True)


def balanced_nll(y, p, subjects):
    losses = -np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1.))
    return float(np.mean([losses[subjects == s].mean() for s in np.unique(subjects)]))


def fit_policy(gru, knn, y, subjects, ids):
    """Inputs must be group-held-out predictions, never base-model training outputs."""
    subjects, y = np.asarray(subjects), np.asarray(y)
    options = []
    for w in PROTOCOL['weight_grid']:
        p = w*gru + (1-w)*knn
        score = np.mean([f1_score(y[subjects == s], p[subjects == s].argmax(1),
                         labels=[0, 1, 2], average='macro', zero_division=0) for s in np.unique(subjects)])
        options.append((float(score), -balanced_nll(y,p,subjects), -abs(w-.5), w))
    weight = max(options)[-1]
    fused = weight*gru + (1-weight)*knn
    optimum = minimize_scalar(lambda logt: balanced_nll(y, temperature(fused, np.exp(logt)), subjects),
                              bounds=(np.log(.25), np.log(10)), method='bounded')
    temp = float(np.exp(optimum.x))
    p = temperature(fused, temp)
    candidates = []
    for threshold in PROTOCOL['confidence_grid']:
        accepted = p.max(1) >= threshold
        if accepted.sum() >= 20 and all(np.sum(accepted & (subjects == s)) >= 5 for s in np.unique(subjects)):
            acc = float(np.mean(p[accepted].argmax(1) == y[accepted]))
            if acc >= .8: candidates.append((int(accepted.sum()), -threshold, threshold, acc))
    choice = max(candidates) if candidates else None
    return dict(gru_weight=weight, temperature=temp,
                confidence_threshold=choice[2] if choice else None,
                threshold_fit_accuracy=choice[3] if choice else None,
                fit_ids=list(map(str, ids)), fit_subjects=sorted(set(map(str, subjects))),
                fit_n=len(y), fit_only_group_oof=True, grid_scores=[list(x) for x in options])


def apply_policy(gru, knn, policy):
    w = policy['gru_weight']
    return temperature(w*gru+(1-w)*knn, policy['temperature'])


def probability_metrics(y, p, subjects, available=None, threshold=None):
    result = metrics(y, p, subjects, available)
    available = np.ones(len(y),bool) if available is None else np.asarray(available,bool)
    valid = available
    result['log_loss_available'] = balanced_nll(y[valid],p[valid],subjects[valid]) if valid.any() else None
    result['brier_available'] = float(np.mean(np.sum((p[valid]-np.eye(3)[y[valid]])**2,axis=1))) if valid.any() else None
    bins=[]; confidence=p.max(1); correct=p.argmax(1)==y
    for low,high in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=valid & (confidence >= low) & ((confidence < high) if high < 1 else (confidence <= high))
        if mask.any(): bins.append(dict(low=float(low),high=float(high),n=int(mask.sum()),confidence=float(confidence[mask].mean()),accuracy=float(correct[mask].mean())))
    result['reliability_bins']=bins
    result['ece_available']=float(sum(b['n']*abs(b['confidence']-b['accuracy']) for b in bins)/max(1,valid.sum()))
    accepted=available & (confidence >= threshold) if threshold is not None else np.zeros(len(y),bool)
    result['selected_threshold']=threshold
    result['selected_coverage']=float(accepted.mean())
    result['selected_accuracy']=float(correct[accepted].mean()) if accepted.any() else None
    result['selected_wrong']=int(np.sum(accepted & ~correct))
    cm=result['confusion_with_unavailable']
    result['class_recall']=[float(cm[i][i]/sum(cm[i])) if sum(cm[i]) else None for i in range(3)]
    return result
