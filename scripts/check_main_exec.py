"""Smoke-test main.py the way kaggle_environments will: exec the source
without populating __file__ in the namespace, from a working directory
that mirrors /kaggle_simulations/agent/.

This catches the failure mode that bit submission 53776705
(NameError: name '__file__' is not defined). The existing
scripts/check_main.py uses importlib which DOES set __file__, so the
bug slipped through the normal pre-commit gate.

Steps:
  1. Build submission.tar.gz via make_submission.sh (uses /tmp output).
  2. Extract into a fresh sandbox directory.
  3. chdir into the sandbox so relative paths ("./deck.csv", "./train")
     resolve like they would at submission time.
  4. Read main.py and exec(compile(src, 'main.py', 'exec'), env) with
     a clean namespace — same call shape as
     kaggle_environments.agents.agent.get_last_callable.
  5. Verify the agent function still loads policies and returns a
     60-card list for the initial deck-submission step.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a submission tar.gz by exec()-loading main.py the same way Kaggle does.",
    )
    parser.add_argument(
        "--tar-gz",
        type=Path,
        default=None,
        help=(
            "Path to submission tar.gz to verify. If omitted, builds a fresh "
            "one with make_submission.sh into a temp directory."
        ),
    )
    parser.add_argument(
        "--no-policy",
        action="store_true",
        help=(
            "Skip the _POLICY presence check. Use for non-MLP bundles "
            "(e.g. rule-based submissions) where agent doesn't expose _POLICY."
        ),
    )
    parser.add_argument(
        "--strict-cwd",
        action="store_true",
        help=(
            "Simulate Kaggle runtime: do NOT chdir into the sandbox, and do "
            "NOT add it to sys.path. main.py must resolve deck.csv / train/ "
            "from absolute paths or via __file__-based discovery. This "
            "catches the failure mode that produced 53810836 / 53812115 ERRORs."
        ),
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="poke_ai_exec_test_") as tmp:
        tmpdir = Path(tmp)
        sandbox = tmpdir / "agent"
        sandbox.mkdir()

        if args.tar_gz is not None:
            tarball = args.tar_gz.resolve()
            if not tarball.exists():
                print(f"--tar-gz path does not exist: {tarball}", file=sys.stderr)
                return 1
        else:
            tarball = tmpdir / "submission.tar.gz"
            # Build the submission. Reuse make_submission.sh so we test the
            # exact bundle layout that goes to Kaggle.
            rc = subprocess.run(
                [str(ROOT / "make_submission.sh"), str(tarball)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            if rc.returncode != 0:
                print(f"make_submission.sh failed:\n{rc.stderr}", file=sys.stderr)
                return 1

        with tarfile.open(tarball, "r:gz") as t:
            # tar.extractall raises DeprecationWarning on 3.12+ without a
            # filter — be explicit to silence and avoid future breakage.
            t.extractall(sandbox, filter="data")  # noqa: S202

        main_py = sandbox / "main.py"
        if not main_py.exists():
            print("main.py not at top of tarball; sandbox layout wrong", file=sys.stderr)
            return 1

        # Now exec main.py the same way kaggle_environments does:
        #   exec(compile(code, 'main.py', 'exec'), env)
        # env is a fresh dict with no __file__ key.
        original_cwd = os.getcwd()
        original_path = list(sys.path)
        if args.strict_cwd:
            # Kaggle-strict simulation: cwd is some unrelated dir, sandbox
            # is NOT on sys.path. main.py must self-discover its location.
            os.chdir(tmpdir)  # parent of sandbox/, no deck.csv here
        else:
            os.chdir(sandbox)
            # Permissive default: Kaggle-environments local sim adds the
            # bundle dir to sys.path. Strict mode skips this.
            sys.path.insert(0, str(sandbox))
        try:
            code = main_py.read_text()
            ns: dict = {}
            compiled = compile(code, "main.py", "exec")
            try:
                exec(compiled, ns)  # noqa: S102
            except Exception as exc:  # noqa: BLE001
                print(
                    f"main.py exec failed (this is the Kaggle failure mode!):\n  {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return 1

            agent = ns.get("agent")
            if not callable(agent):
                print("after exec, agent is not callable in the namespace", file=sys.stderr)
                return 1

            if args.no_policy:
                print("  exec OK; agent loaded (--no-policy: _POLICY check skipped)")
            else:
                policy = ns.get("_POLICY")
                if policy is None:
                    print(
                        "after exec, _POLICY is None — bundle is missing the MLP and "
                        "the linear fallback. Submission would run on the engine-prior "
                        "baseline only.",
                        file=sys.stderr,
                    )
                    return 1
                print(f"  exec OK; agent loaded, _POLICY type = {type(policy).__name__}")

            deck = agent({"select": None, "logs": [], "current": None})
            if not isinstance(deck, list) or len(deck) != 60:
                print(
                    f"after exec, agent returned {type(deck).__name__} of len "
                    f"{len(deck) if hasattr(deck, '__len__') else '?'} (expected list of 60)",
                    file=sys.stderr,
                )
                return 1
            if not all(isinstance(x, int) and x > 0 for x in deck):
                print(
                    "after exec, deck contains non-positive-int entries",
                    file=sys.stderr,
                )
                return 1
            print(f"  agent({{select=None}}) returned 60 cards, first={deck[:3]}...")
        finally:
            os.chdir(original_cwd)
            sys.path[:] = original_path

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
