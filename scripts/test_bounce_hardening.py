import sys
import numpy as np
sys.path.insert(0, r"C:\Users\hibye\Documents\ACTUAL PROJECTS\Thesis\scripts")
from balltrack_pipeline import detect_bounce
from live_app import smooth_flip


def mk(n, pts, wpts):
    b = np.full((n, 2), np.nan)
    w = np.full((n, 2), np.nan)
    for f, (x, y) in pts.items():
        b[f] = (x, y)
    for f, (x, y) in wpts.items():
        w[f] = (x, y)
    return b, w


n = 900
pts = {}
wpts = {}

for f in range(0, 41):
    pts[f] = (337.0, 294.0)
    wpts[f] = (-75.0, -80.0)

for f in range(161, 189):
    pts[f] = (1420.0, 470.0 + (f - 161) * 3.9)
    wpts[f] = (19.0, 30.0)

for f in range(191, 202):
    pts[f] = (337.0, 301.6)
    wpts[f] = (-75.0, -80.0)

for f in range(204, 235):
    t = f - 204
    pts[f] = (1300.0 - 8 * t, 533.0 - 6 * t + 0.1 * t * t)
    wpts[f] = (17.0, 31.0)

b, w = mk(n, pts, wpts)
r = detect_bounce(b, w, min_i=29, fps=59.94)
print("A static+ramp rejected, real bounce found:", r)
assert r is not None and 185 <= r[0] <= 200, r

pts2 = {}
wpts2 = {}
for f in range(0, 30):
    pts2[f] = (337.0, 294.0)
    wpts2[f] = (-75.0, -80.0)
for f in range(100, 121):
    pts2[f] = (700.0, 400.0 + (f - 100) * 12.0)
    wpts2[f] = (5.0, 20.0)
for f in range(150, 161):
    pts2[f] = (694.0, 220.0)
    wpts2[f] = (6.0, 8.0)

b2, w2 = mk(n, pts2, wpts2)
r2 = detect_bounce(b2, w2, min_i=29, fps=59.94)
print("B fall then stale cluster across gap:", r2)
assert r2 is None, r2

b3 = np.full((150, 2), np.nan)
w3 = np.full((150, 2), np.nan)
for f in range(0, 35):
    b3[f] = (337.0, 294.0)
    w3[f] = (-75.0, -80.0)
for f in range(90, 115):
    t = f - 90
    b3[f] = (700.0, 300.0 + (12 - abs(t - 12)) ** 2 * 1.5)
    w3[f] = (8.0, 25.0)
for f in range(120, 150):
    b3[f] = (337.0, 301.6)
    w3[f] = (-75.0, -80.0)

r3 = smooth_flip(b3[:, 1], w3[:, 0], w3[:, 1], fps=59.94)
print("C live window bounce:", r3)
assert r3 is not None and 90 <= r3[0] <= 130, r3

b4 = np.full((150, 2), np.nan)
w4 = np.full((150, 2), np.nan)
for f in range(0, 60):
    b4[f] = (337.0, 294.0)
    w4[f] = (-75.0, -80.0)
r4 = smooth_flip(b4[:, 1], w4[:, 0], w4[:, 1], fps=59.94)
print("D all-static window:", r4)
assert r4 is None, r4

print("ALL PASS")
