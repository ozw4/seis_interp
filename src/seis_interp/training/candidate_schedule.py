"""Candidate priority and isolated CPU processes, without changing selection order."""

import multiprocessing
from contextlib import suppress
from itertools import islice
from multiprocessing.connection import wait

from seis_interp.training.run_recovery import RestartRunProcess
from seis_interp.training.worker_processes import process_start_time, source_root


def ordered_candidates(candidates, priority):
    ids = [item["id"] for item in candidates]
    if len(set(priority)) != len(priority) or set(priority) - set(ids):
        raise ValueError("candidate_priority must contain distinct existing configuration IDs")
    ranks = {name: index for index, name in enumerate(priority)}
    return sorted(candidates, key=lambda item: ranks.get(item["id"], len(priority)))


def _execute_child(connection, execute, candidate):
    try:
        connection.send(("result", execute(candidate)))
    except BaseException as error:
        connection.send(("error", error))
    finally:
        connection.close()


def candidate_outcomes(execute, candidates, parallel=1, *, children_changed=None, failed=None):
    """One process per concurrent run: an abrupt exit never breaks a sibling pool.

    Restart just the exited job in a fresh process, at most twice. The run wrapper
    persists its own attempt budget across this scheduler and supervisor restarts.
    Ordinary errors stop submission and let all already-running siblings finish.
    """
    if isinstance(parallel, bool) or not isinstance(parallel, int) or parallel < 1:
        raise ValueError("parallel_candidates must be a positive integer")
    if parallel == 1:
        for candidate in candidates:
            yield execute(candidate)
        return
    context = multiprocessing.get_context("spawn")
    remaining = iter(candidates)
    active = {}

    def publish_children():
        if children_changed is not None:
            children_changed(
                [
                    {
                        "pid": process.pid,
                        "process_start_time": process_start_time(process.pid),
                        "source_root": source_root(),
                        "run": f"seed_{candidate['seed']}"
                        if "seed" in candidate
                        else candidate["id"],
                        "restarts": restarts,
                    }
                    for process, candidate, restarts in active.values()
                ]
            )

    def start(candidate, restarts=0):
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_execute_child, args=(sender, execute, candidate))
        process.start()
        sender.close()
        active[receiver] = process, candidate, restarts
        publish_children()

    try:
        for candidate in islice(remaining, parallel):
            start(candidate)
        while active:
            ready = wait(list(active))
            results = []
            for connection in ready:
                process, candidate, restarts = active.pop(connection)
                try:
                    kind, value = connection.recv()
                except EOFError:
                    kind, value = "restart", RuntimeError("worker exited during an isolated run")
                finally:
                    connection.close()
                process.join()
                process.close()
                if kind == "result":
                    results.append(value)
                elif (kind == "restart" or isinstance(value, RestartRunProcess)) and restarts < 2:
                    start(candidate, restarts + 1)
                else:
                    raise value
            publish_children()
            yield from results
            # Drain every completed job before scheduling new candidates.
            if active and wait(list(active), timeout=0):
                continue
            for candidate in islice(remaining, parallel - len(active)):
                start(candidate)
    except BaseException as error:
        if failed is not None:
            failed(error)
        raise
    finally:
        for connection, (process, _, _) in active.items():
            # Drain the pipe before join so a large result cannot block its owner.
            with suppress(EOFError):
                connection.recv()
            connection.close()
            process.join()
            process.close()
        if children_changed is not None:
            children_changed([])
