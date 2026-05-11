"""``python -m agent_kernel`` dispatcher.

Forwards to the CLI Click group so ``python -m agent_kernel install --user``
works the same as ``agent-kernel install --user``.
"""

from agent_kernel.cli import main

if __name__ == "__main__":  # pragma: no cover
    main()
