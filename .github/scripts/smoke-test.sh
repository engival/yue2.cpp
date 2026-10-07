#!/usr/bin/env bash
# CI smoke test for a yue2 build: no GPU, no model weights.
#
#	.github/scripts/smoke-test.sh BUILD_DIR
#
# - `yue2` with no arguments prints its usage and exits 1; `yue2 song --help` exits 0.
# - `yue2 noise` reproduces the golden of tests/golden/noise_meta.json byte for byte
#   (set NOISE_GOLDEN_SOFT=1 to downgrade a mismatch to a warning).
# - Every tests-only target that needs no model (YUE2_BUILD_TESTS=ON) is run when
#   it was built; one that is missing is reported and skipped.
# Runs under Git Bash on Windows too.

set -euo pipefail

build=${1:?usage: smoke-test.sh BUILD_DIR}
exe=""
case "$(uname -s)" in
	MINGW*|MSYS*|CYGWIN*) exe=".exe" ;;
esac
yue2="$build/yue2$exe"
work=$(mktemp -d "$build/smoke.XXXXXX")
failed=0

fail()
{
	echo "::error::$*"
	failed=1
}

echo "== yue2 (no arguments)"
set +e
"$yue2" > "$work/noargs.out" 2> "$work/noargs.err"
rc=$?
set -e
cat "$work/noargs.err"
if [ "$rc" -ne 1 ]; then fail "yue2 with no arguments exited $rc, expected 1"; fi
if ! grep -q "usage: yue2 song" "$work/noargs.err"; then fail "yue2 with no arguments printed no usage"; fi

echo "== yue2 song --help"
if ! "$yue2" song --help; then fail "yue2 song --help exited non-zero"; fi

echo "== yue2 noise vs tests/golden/noise_meta.json"
meta=tests/golden/noise_meta.json
want=$(grep -o '"sha256": *"[0-9a-f]*"' "$meta" | grep -o '[0-9a-f]\{64\}')
seed=$(grep -o '"seed": *[0-9]*' "$meta" | grep -o '[0-9]*$')
frames=$(grep -o '"shape": *\[ *[0-9]*' "$meta" | grep -o '[0-9]*$')
"$yue2" noise --seed "$seed" --frames "$frames" -o "$work/noise.npy"
got=$(sha256sum "$work/noise.npy" | cut -d' ' -f1)
echo "want $want"
echo "got  $got"
if [ "$got" != "$want" ]
then
	if [ "${NOISE_GOLDEN_SOFT:-0}" = "1" ]
	then
		echo "::warning::yue2 noise output differs from the golden on this platform (libm sqrt/log/cos/sin)"
	else
		fail "yue2 noise output differs from tests/golden/noise_meta.json"
	fi
fi

# Tests-only targets that need no model and no device. yue2-sampler-diff is
# trimmed from its 100k default cases to keep CI short.
run_test()
{
	local name=$1
	shift
	local bin="$build/$name$exe"
	if [ ! -x "$bin" ]
	then
		echo "== $name: not built, skipped"
		return
	fi
	echo "== $name $*"
	local t0=$SECONDS
	if ! "$bin" "$@"; then fail "$name failed"; fi
	echo "   ($((SECONDS - t0)) s)"
}

run_test yue2-numpy-exp
run_test yue2-bars
run_test yue2-guidance
run_test yue2-handover
run_test yue2-draft-accept
run_test yue2-sampler-diff --cases 20000

rm -rf -- "$build/smoke.${work##*.}"
exit "$failed"
