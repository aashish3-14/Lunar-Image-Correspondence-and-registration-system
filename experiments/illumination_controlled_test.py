from __future__ import annotations

from pathlib import Path
import json
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lightglue import LightGlue, SuperPoint
from ai_engine.preprocessing import load_grayscale_image, preprocess_image, structural_representation
from ai_engine.spatial import analyze_spatial_distribution

SOURCE = ROOT / 'data' / 'processed' / 'lro_apollo16_overlap.png'
RESULTS = ROOT / 'data' / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

def to_tensor(image, device):
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

def make_variants(image):
    x = image.astype(np.float32) / 255.0
    h, w = image.shape
    field = np.tile(np.linspace(0.65, 1.35, w, dtype=np.float32)[None, :], (h, 1))
    return {
        'darker_0.55x': np.clip(x * 0.55, 0, 1).astype(np.float32),
        'brighter_1.35x': np.clip(x * 1.35, 0, 1).astype(np.float32),
        'gamma_1.8': np.clip(np.power(x, 1.8), 0, 1).astype(np.float32),
        'spatial_gradient': np.clip(x * field, 0, 1).astype(np.float32),
    }

def to_uint8(x):
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)

def evaluate(source, reference, extractor, matcher, device):
    a = to_tensor(source, device)
    b = to_tensor(reference, device)
    with torch.inference_mode():
        f0 = extractor.extract(a)
        f1 = extractor.extract(b)
        out = matcher({'image0': f0, 'image1': f1})
    k0 = f0['keypoints'][0].detach().cpu().numpy()
    k1 = f1['keypoints'][0].detach().cpu().numpy()
    matches = out['matches'][0].detach().cpu().numpy()
    r = {'source_keypoints': int(len(k0)), 'reference_keypoints': int(len(k1)), 'matches': int(len(matches)), 'inliers': 0, 'inlier_ratio': 0.0, 'rmse_px': None, 'spatial_coverage': 0.0, 'spatial_uniformity': 0.0}
    if len(matches) < 4:
        return r
    p0, p1 = k0[matches[:, 0]], k1[matches[:, 1]]
    H, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return r
    mask = mask.ravel().astype(bool)
    ie = np.linalg.norm(cv2.perspectiveTransform(p0.reshape(-1,1,2).astype(np.float32), H).reshape(-1,2) - p1, axis=1)[mask]
    spatial = analyze_spatial_distribution(p0[mask], source.shape[1], source.shape[0], 4, 4)
    r.update({'inliers': int(mask.sum()), 'inlier_ratio': float(mask.mean()), 'rmse_px': float(np.sqrt(np.mean(ie ** 2))) if len(ie) else None, 'spatial_coverage': float(spatial['coverage']), 'spatial_uniformity': float(spatial['uniformity'])})
    return r

def show(name, r):
    print(f'  {name}: keypoints={r["source_keypoints"]}/{r["reference_keypoints"]}, matches={r["matches"]}, inliers={r["inliers"]}, ratio={r["inlier_ratio"]:.3f}, RMSE={r["rmse_px"]}, coverage={r["spatial_coverage"]:.3f}, uniformity={r["spatial_uniformity"]:.3f}')

def main():
    print('=' * 72)
    print('LUCAS | STEP 1B | CONTROLLED ILLUMINATION EXPERIMENT')
    print('=' * 72)
    base = load_grayscale_image(SOURCE)
    variants = make_variants(base)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda': print(f'GPU: {torch.cuda.get_device_name(0)}')
    extractor = SuperPoint(max_num_keypoints=4096).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)
    methods = {'existing': preprocess_image, 'structural': structural_representation}
    results = {}
    for condition, image in [('original', base)] + list(variants.items()):
        print('\n' + '-' * 72)
        print(f'CONDITION: {condition}')
        print('-' * 72)
        results[condition] = {}
        for method, fn in methods.items():
            src = fn(to_uint8(image) if image.dtype != np.uint8 else image)
            ref = fn(base)
            results[condition][method] = evaluate(src, ref, extractor, matcher, device)
            show(method, results[condition][method])
    path = RESULTS / 'illumination_controlled_results.json'
    path.write_text(json.dumps(results, indent=4), encoding='utf-8')
    print('\nSaved:', path)
    print('STEP 1B COMPLETE')

if __name__ == '__main__':
    main()
