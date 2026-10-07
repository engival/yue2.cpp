// Speculative acceptance (SPEC_DRAFT.md §4): the draw, the accept test and the
// residual on small sparse distributions. No model and no device, so
// tests/draft_accept.cpp checks exactly this code.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

// The first i with r < w[0] + ... + w[i]; the last index when rounding leaves r
// at or past the sum. sample_step's draw since stage 5, unchanged.
inline size_t inverse_cdf(const double * w, size_t n, double r)
{
	double c = 0;
	for (size_t i = 0; i < n; i++)
	{
		c += w[i];
		if (r < c)
		{
			return i;
		}
	}
	return n - 1;
}

// One distribution over token ids. Sparse: `id[i]` has `prob[i]`, in the
// sampler's order (p descending, id ascending on ties). Dense (`base` >= 0):
// `prob[i]` belongs to id base + i and `id` is unused — the raw draft q, which
// covers the whole semantic vocabulary.
struct DraftDist
{
	std::vector<int>    id;
	std::vector<double> prob;
	int                 base = -1;

	int at(size_t i) const
	{
		return base >= 0 ? base + (int) i : id[i];
	}

	double of(int x) const
	{
		if (base >= 0)
		{
			return x >= base && x - base < (int) prob.size() ? prob[(size_t) (x - base)] : 0.0;
		}
		for (size_t i = 0; i < id.size(); i++)
		{
			if (id[i] == x)
			{
				return prob[i];
			}
		}
		return 0.0;
	}

	int draw(double u) const
	{
		return at(inverse_cdf(prob.data(), prob.size(), u));
	}
};

// Accept a drafted token with probability min(1, λ·p/q) (§3.3 step 3, §4.3).
// p == 0 — a token the target could not have drawn — is always rejected. Written
// without the divide: a raw q's rounding fallback can draw a token whose q
// underflowed to 0.
inline bool draft_accept(double u, double lambda, double p, double q)
{
	return p > 0 && u * q < lambda * p;
}

// The replacement after a reject: r(x) ∝ max(0, p(x) − q(x)) over p's
// candidates, which hold all of r's support, in p's order and with one uniform.
// When r has no mass (p == q to rounding) the draw is from p itself (§4.1).
// `w` is scratch.
inline int draft_residual(const DraftDist & p, const DraftDist & q, double u, std::vector<double> & w)
{
	w.resize(p.prob.size());
	double sum = 0;
	for (size_t i = 0; i < p.prob.size(); i++)
	{
		w[i] = std::max(0.0, p.prob[i] - q.of(p.at(i)));
		sum += w[i];
	}
	if (sum <= 1e-12)
	{
		return p.draw(u);
	}
	// inverse_cdf's fallback is the last candidate, which may carry no residual
	// mass: step back to one that does.
	size_t i = inverse_cdf(w.data(), w.size(), u * sum);
	while (w[i] == 0 && i > 0)
	{
		i--;
	}
	return p.at(i);
}

// Σ_x min(p(x), q(x)) = 1 − TV(p, q): the acceptance a depth has at λ = 1.
inline double draft_overlap(const DraftDist & p, const DraftDist & q)
{
	double sum = 0;
	for (size_t i = 0; i < p.prob.size(); i++)
	{
		sum += std::min(p.prob[i], q.of(p.at(i)));
	}
	return sum;
}

// Shannon entropy in nats.
inline double draft_entropy(const DraftDist & p)
{
	double h = 0;
	for (size_t i = 0; i < p.prob.size(); i++)
	{
		h -= p.prob[i] > 0 ? p.prob[i] * std::log(p.prob[i]) : 0.0;
	}
	return h;
}
