#!/usr/bin/env python3
# trace_summary.py — read a --guidance-trace file as a timeline. Needs numpy only.
#
#   python3 scripts/trace_summary.py DIR/guidance_trace.npy [--every SECONDS] [--top N]
#
# One row per SECONDS of audio (default 5): the weights in force, how far each branch stood from
# the positive one (tv_prev / tv_blank), how far the blend moved what the sampler saw (push, with a
# bar), how often `previous` still wanted the same token, and the mean log p of the drawn tokens
# under the UNGUIDED model. Then the N hardest-pushed moments as timestamps to seek to.
# It describes what the guidance did; whether it sounds good is for your ears.
import argparse, warnings, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('trace'); ap.add_argument('--every', type=float, default=5); ap.add_argument('--top', type=int, default=10)
a = ap.parse_args()
warnings.simplefilter('ignore', RuntimeWarning)		# a bin where a branch is not live is all NaN

def clock(step):
	return '%d:%04.1f' % (step // 1500, step % 1500 / 25)

t = np.load(a.trace)
if t.ndim != 2 or t.shape[1] != 8 or not len(t):
	raise SystemExit('%s: expected a non-empty [steps, 8] float32 trace' % a.trace)
print('%d traced steps, %s - %s' % (len(t), clock(t[0, 0]), clock(t[-1, 0])))
print('%-8s %6s %6s  %7s %8s  %5s %-20s  %5s  %6s' % ('from', 'w_prev', 'w_blnk', 'tv_prev', 'tv_blank', 'push', '', 'same', 'logp'))
for b in np.unique(t[:, 0] // (a.every * 25)):
	r = t[t[:, 0] // (a.every * 25) == b]
	print('%-8s %6.2f %6.2f  %7.2f %8.2f  %5.2f %-20s  %4.0f%%  %6.2f' % (clock(r[0, 0]), r[:, 1].mean(), r[:, 2].mean(),
		np.nanmean(r[:, 3]), np.nanmean(r[:, 4]), r[:, 6].mean(), '#' * round(20 * r[:, 6].mean()), 100 * np.nanmean(r[:, 5]), r[:, 7].mean()))
print('\nhardest pushes (1 s windows; push = mean tv_blend, p = what the unguided model gave the drawn tokens):')
sec = np.unique(t[:, 0] // 25)
push = np.array([t[t[:, 0] // 25 == s, 6].mean() for s in sec])
for i in sorted(np.argsort(-push)[:a.top]):
	print('  %-8s push %.2f   p %.3f' % (clock(sec[i] * 25), push[i], np.exp(t[t[:, 0] // 25 == sec[i], 7].mean())))
