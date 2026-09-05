from rebuno._kernel import DispatchLease
from rebuno.execution import ExecutionContext, _set_current


class FakeKernel:
    def __init__(self, decision):
        self.decision = decision
        self.completed = []

    async def submit_step(
        self, execution_id, *, lease, kind, target, args, idempotency
    ):
        self.captured = dict(
            kind=kind, target=target, args=args, idempotency=idempotency
        )
        return self.decision

    async def complete_step(self, execution_id, step_id, *, lease, result):
        self.completed.append(result)


def install_context(kernel):
    return _set_current(
        ExecutionContext(
            kernel=kernel,
            execution_id="e1",
            lease=DispatchLease("d1", 1, 120.0),
            agent_id="a",
            input=None,
        )
    )
