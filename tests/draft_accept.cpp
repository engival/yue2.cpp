// yue2-draft-accept — SPEC_DRAFT.md §4.2: the speculative accept/residual math
// of src/draft_accept.hpp on random sparse (p, q) pairs, 10^6 rounds a case.
// One depth is enough: every depth of a round is this step on its own pair.
//
// λ = 1: the emitted tokens must be distributed as p (chi-square, α = 1e-4).
// λ > 1: they must stay inside supp(p) and match §4.3's law
//        out(x) = min(q, λp)·[x ∈ supp p] + (1 − A)·r(x); KL(out‖p) is printed.
//
//	build_draft/yue2-draft-accept [-v]
//
// Exit 0 = every case passed.

#include "draft_accept.hpp"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace
{

const int    DRAWS   = 1000000;
const int    UNIVERSE = 400;     // ids are drawn from [0, UNIVERSE)
const double Z_ALPHA = 3.719;    // one-sided normal quantile for α = 1e-4

int  failures = 0;
int  checked  = 0;
bool verbose  = false;

// A sparse distribution over `ids` with random weights, in the sampler's order
// (p descending, id ascending on ties).
DraftDist sparse(const std::vector<int> & ids, std::mt19937_64 & rng, double skew)
{
	std::exponential_distribution<double> ex(1.0);
	std::vector<std::pair<double, int>>   w;
	double                                sum = 0;
	for (size_t i = 0; i < ids.size(); i++)
	{
		const double v = std::pow(ex(rng), skew);
		w.push_back(std::make_pair(v, ids[i]));
		sum += v;
	}
	std::sort(w.begin(), w.end(), [](const std::pair<double, int> & a, const std::pair<double, int> & b)
	{
		return a.first != b.first ? a.first > b.first : a.second < b.second;
	});
	DraftDist d;
	for (size_t i = 0; i < w.size(); i++)
	{
		d.id.push_back(w[i].second);
		d.prob.push_back(w[i].first / sum);
	}
	return d;
}

std::vector<int> range(int from, int to)
{
	std::vector<int> out;
	for (int i = from; i < to; i++)
	{
		out.push_back(i);
	}
	return out;
}

// §4.3's emitted-token law; at λ = 1 it is p.
std::vector<double> law(const DraftDist & p, const DraftDist & q, double lambda)
{
	std::vector<double> out(UNIVERSE, 0.0);
	double a = 0;
	for (int x = 0; x < UNIVERSE; x++)
	{
		const double m = std::min(q.of(x), lambda * p.of(x));
		out[(size_t) x] = m;
		a += m;
	}
	double rsum = 0;
	for (size_t i = 0; i < p.prob.size(); i++)
	{
		rsum += std::max(0.0, p.prob[i] - q.of(p.at(i)));
	}
	for (size_t i = 0; i < p.prob.size(); i++)
	{
		const double r = rsum <= 1e-12 ? p.prob[i] : std::max(0.0, p.prob[i] - q.of(p.at(i))) / rsum;
		out[(size_t) p.at(i)] += (1 - a) * r;
	}
	return out;
}

// One case: DRAWS single-depth rounds, exactly as Runner::draft_round runs a
// depth — d ~ q, accept with one uniform, else the residual with one more.
void run_case(const char * name, const DraftDist & p, const DraftDist & q, double lambda, uint64_t seed)
{
	std::mt19937_64                        rng(seed);
	std::uniform_real_distribution<double> uniform(0.0, 1.0);
	std::vector<double>                    w;
	std::vector<long long>                 count(UNIVERSE, 0);
	long long                              accepted = 0;
	for (int n = 0; n < DRAWS; n++)
	{
		const int d = q.draw(uniform(rng));
		if (draft_accept(uniform(rng), lambda, p.of(d), q.of(d)))
		{
			count[(size_t) d]++;
			accepted++;
			continue;
		}
		count[(size_t) draft_residual(p, q, uniform(rng), w)]++;
	}

	// Outside supp(p) is a failure at any λ.
	long long outside = 0;
	for (int x = 0; x < UNIVERSE; x++)
	{
		outside += p.of(x) == 0 ? count[(size_t) x] : 0;
	}

	// Chi-square against the law, bins under 5 expected pooled into one.
	const std::vector<double> want = law(p, q, lambda);
	double chi2 = 0, pool_obs = 0, pool_exp = 0, kl = 0;
	int    bins = 0;
	for (int x = 0; x < UNIVERSE; x++)
	{
		const double e = want[(size_t) x] * DRAWS;
		const double o = (double) count[(size_t) x];
		if (o > 0 && p.of(x) > 0)
		{
			kl += o / DRAWS * std::log(o / DRAWS / p.of(x));
		}
		if (e < 5)
		{
			pool_obs += o;
			pool_exp += e;
			continue;
		}
		chi2 += (o - e) * (o - e) / e;
		bins++;
	}
	if (pool_exp >= 5)
	{
		chi2 += (pool_obs - pool_exp) * (pool_obs - pool_exp) / pool_exp;
		bins++;
	}
	// Wilson–Hilferty: the chi-square quantile at 1 − α for df degrees of freedom.
	const int    df   = std::max(1, bins - 1);
	const double c    = 2.0 / (9.0 * df);
	const double crit = df * std::pow(1 - c + Z_ALPHA * std::sqrt(c), 3);
	const bool   ok   = outside == 0 && chi2 <= crit;

	checked++;
	failures += ok ? 0 : 1;
	if (verbose || !ok)
	{
		printf("%s %-28s λ %.1f  |p| %3zu |q| %3zu  accept %.4f  overlap %.4f  chi2 %8.1f / %8.1f (df %d)  "
		       "outside %lld  KL(out‖p) %.5f\n", ok ? "ok  " : "FAIL", name, lambda,
		       p.prob.size(), q.base >= 0 ? q.prob.size() : q.id.size(),
		       (double) accepted / DRAWS, draft_overlap(p, q), chi2, crit, df, outside, kl);
	}
}

void check(bool ok, const char * what)
{
	checked++;
	failures += ok ? 0 : 1;
	if (verbose || !ok)
	{
		printf("%s %s\n", ok ? "ok  " : "FAIL", what);
	}
}

} // namespace

int main(int argc, char ** argv)
{
	for (int i = 1; i < argc; i++)
	{
		verbose = verbose || strcmp(argv[i], "-v") == 0;
	}

	std::mt19937_64 rng(20261006);

	struct Pair
	{
		const char * name;
		DraftDist    p;
		DraftDist    q;
	};
	std::vector<Pair> pairs;

	pairs.push_back({ "disjoint supports", sparse(range(0, 100), rng, 2), sparse(range(100, 200), rng, 2) });
	{
		const DraftDist p = sparse(range(0, 60), rng, 2);
		pairs.push_back({ "q superset of p", p, sparse(range(0, 100), rng, 2) });
		pairs.push_back({ "p superset of q", sparse(range(0, 100), rng, 2), sparse(range(20, 50), rng, 2) });
	}
	pairs.push_back({ "single-id p, inside q", sparse({ 7 }, rng, 1), sparse(range(0, 100), rng, 2) });
	pairs.push_back({ "single-id p, outside q", sparse({ 300 }, rng, 1), sparse(range(0, 100), rng, 2) });
	{
		// p ≈ q: q is p with every weight nudged by a few parts in 1e9.
		const DraftDist p = sparse(range(0, 100), rng, 2);
		DraftDist       q = p;
		double          sum = 0;
		for (size_t i = 0; i < q.prob.size(); i++)
		{
			q.prob[i] *= 1 + 3e-9 * (double) (i % 3);
			sum       += q.prob[i];
		}
		for (size_t i = 0; i < q.prob.size(); i++)
		{
			q.prob[i] /= sum;
		}
		pairs.push_back({ "near-equal p and q", p, q });
		pairs.push_back({ "identical p and q", p, p });
	}
	pairs.push_back({ "partial overlap, peaked", sparse(range(0, 100), rng, 4), sparse(range(50, 150), rng, 4) });
	{
		// --draft-q raw: q dense over the whole universe.
		DraftDist q = sparse(range(0, UNIVERSE), rng, 2);
		DraftDist dense;
		dense.base = 0;
		dense.prob.assign(UNIVERSE, 0.0);
		for (size_t i = 0; i < q.id.size(); i++)
		{
			dense.prob[(size_t) q.id[i]] = q.prob[i];
		}
		pairs.push_back({ "dense q (raw)", sparse(range(100, 200), rng, 2), dense });
	}

	const double lambdas[] = { 1.0, 1.5, 2.0, 3.0 };
	for (size_t i = 0; i < pairs.size(); i++)
	{
		for (size_t l = 0; l < sizeof(lambdas) / sizeof(lambdas[0]); l++)
		{
			run_case(pairs[i].name, pairs[i].p, pairs[i].q, lambdas[l], 1000 * i + l + 1);
		}
	}

	// The edges, stated directly.
	{
		const DraftDist p = sparse(range(0, 10), rng, 1);
		std::vector<double> w;
		check(!draft_accept(0.0, 1.0, 0.0, 0.5), "p(d) = 0 is always rejected, even at u = 0");
		check(!draft_accept(0.0, 3.0, 0.0, 0.5), "and at λ = 3");
		check(draft_accept(0.999999, 1.0, 0.5, 0.25), "p > q is always accepted");
		check(draft_residual(p, p, 0.0, w) == p.at(0), "p == q falls back to p (u = 0: p's head)");
		check(draft_residual(p, p, 0.999999999, w) == p.at(p.prob.size() - 1),
		      "p == q falls back to p (u → 1: p's tail)");
	}

	printf("%s: %d cases, %d failures\n", failures == 0 ? "PASS" : "FAIL", checked, failures);
	return failures == 0 ? 0 : 1;
}
