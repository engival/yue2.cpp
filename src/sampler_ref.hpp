// The stage-5 `sample_step`, frozen verbatim as `sample_step_ref`.
//
// This is the reference the stage-6 sparse sampler is proved equal to
// (SPEC_SAMPLER.md §4): `yue2 ar --verify-sampler` runs both per step and dies
// on the first disagreement, and `yue2-sampler-diff` drives both over synthetic
// logits. It is dead weight otherwise.
//
// DO NOT EDIT. Its only value is being the code that shipped; a "fix" here
// silently weakens every equivalence claim in src/STATUS_SAMPLER.md.
//
// Included from stage_ar.cpp *inside* its anonymous namespace, after the
// protocol constants and `Sampling` — it uses both and deliberately owns no
// copy of them, so the two implementations can never drift apart on the
// vocabulary layout or the generation config.
#pragma once

static llama_token sample_step_ref(const float * logits, int n_vocab, const Sampling & s,
	const std::vector<llama_token> & history, int step,
	bool phase_abc, bool legacy_off, std::mt19937_64 & rng)
{
	const float ninf = -std::numeric_limits<float>::infinity();
	const int   end  = phase_abc ? ABC_END : MUSIC_END;

	std::vector<float> scores(logits, logits + n_vocab);

	// allowed mask
	if (phase_abc)
	{
		for (int i = EOD; i < n_vocab; i++)
		{
			scores[i] = ninf;
		}
	} else {
		for (int i = 0; i < CODEC_OFFSET && i < n_vocab; i++)
		{
			scores[i] = ninf;
		}
		for (int i = CODEC_OFFSET + CODEC_SIZE; i < n_vocab; i++)
		{
			scores[i] = ninf;
		}
	}
	scores[end] = logits[end];

	if (step < s.min_tokens)
	{
		scores[end] = ninf;
	}

	// window penalty over the last penalty_window generated ids
	if (s.repetition_penalty != 1.0 && !history.empty())
	{
		const size_t from = history.size() > (size_t) s.penalty_window
		                    ? history.size() - (size_t) s.penalty_window : 0;
		std::unordered_map<llama_token, int> freq;
		for (size_t i = from; i < history.size(); i++)
		{
			freq[history[i]]++;
		}
		for (const auto & kv : freq)
		{
			if (kv.first < 0 || kv.first >= n_vocab)
			{
				continue;
			}
			const float alpha = (float) std::pow(s.repetition_penalty, (double) kv.second);
			float &     v     = scores[kv.first];
			v = v < 0 ? v * alpha : v / alpha;
		}
	}

	// greedy: argmax over the penalised scores, first index on a tie (torch semantics)
	if (s.temperature == 0)
	{
		int   best = 0;
		float top  = ninf;
		for (int i = 0; i < n_vocab; i++)
		{
			if (scores[i] > top)
			{
				top  = scores[i];
				best = i;
			}
		}
		return (llama_token) best;
	}

	if (s.temperature != 1)
	{
		// float division, like torch's float32 tensor / float scalar
		const float t = (float) s.temperature;
		for (int i = 0; i < n_vocab; i++)
		{
			scores[i] /= t;
		}
	}

	// candidates = everything still finite. The allowed set is one contiguous
	// range plus `end`, so that is the exact reservation.
	std::vector<std::pair<float, int>> cand;
	cand.reserve((size_t) (phase_abc ? EOD : CODEC_SIZE) + 1);
	for (int i = 0; i < n_vocab; i++)
	{
		if (scores[i] > ninf)
		{
			cand.push_back(std::make_pair(scores[i], i));
		}
	}
	if (cand.empty())
	{
		die("every token was masked out (step %d)", step);
	}

	// top-k: keep everything >= the k-th largest value
	if ((int) cand.size() > s.top_k)
	{
		std::nth_element(cand.begin(), cand.begin() + (s.top_k - 1), cand.end(), by_score_desc);
		const float threshold = cand[(size_t) s.top_k - 1].first;
		std::vector<std::pair<float, int>> kept;
		kept.reserve((size_t) s.top_k + 8);
		for (size_t i = 0; i < cand.size(); i++)
		{
			if (!(cand[i].first < threshold))
			{
				kept.push_back(cand[i]);
			}
		}
		cand.swap(kept);
	}

	std::sort(cand.begin(), cand.end(), by_score_then_id);

	// softmax over the surviving scores
	std::vector<double> prob(cand.size());
	{
		const double top = cand[0].first;
		double       sum = 0;
		for (size_t i = 0; i < cand.size(); i++)
		{
			prob[i] = std::exp((double) cand[i].first - top);
			sum    += prob[i];
		}
		for (size_t i = 0; i < prob.size(); i++)
		{
			prob[i] /= sum;
		}
	}

	// top-p: drop where cumsum - p > top_p, always keeping the head
	if (s.top_p < 1)
	{
		const size_t keep_head = legacy_off ? 3 : 1;
		double       cum       = 0;
		size_t       n         = cand.size();
		for (size_t i = 0; i < cand.size(); i++)
		{
			cum += prob[i];
			if (i >= keep_head && cum - prob[i] > s.top_p)
			{
				n = i;
				break;
			}
		}
		cand.resize(n);
		prob.resize(n);
		double sum = 0;
		for (size_t i = 0; i < prob.size(); i++)
		{
			sum += prob[i];
		}
		for (size_t i = 0; i < prob.size(); i++)
		{
			prob[i] /= sum;
		}
	}

	std::uniform_real_distribution<double> uniform(0.0, 1.0);
	const double                           r = uniform(rng);
	double                                 c = 0;
	for (size_t i = 0; i < prob.size(); i++)
	{
		c += prob[i];
		if (r < c)
		{
			return (llama_token) cand[i].second;
		}
	}
	return (llama_token) cand.back().second;
}
