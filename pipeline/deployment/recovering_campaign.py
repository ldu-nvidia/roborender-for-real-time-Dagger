"""CPU supervisor: monitor genuine allocation and retry only safe round startup.

No native restart, arbitrary process kill, unreserved fallback or duplicate
optimizer submission. A lost/uncertain in-window owner is terminal.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from pipeline.deployment.beta_operations import atomic, round_runtime_path


def retryable_boundary(state, failure):
    rid=state.get("round_id")
    return (state.get("phase")=="ABANDONED" and type(rid) is int and rid in (2,3,4)
        and failure.get("phase")=="PREPARE" and failure.get("native_restart_allowed") is False
        and [r.get("round_id") for r in state.get("completed_rounds",[])]==list(range(1,rid)))


def summary(state):
    rows=state.get("completed_rounds",[])
    return {"run_id":state.get("run_id"),"phase":state.get("phase"),"round_id":state.get("round_id"),
        "progress":state.get("progress"),"completed_rounds":[{
            "round_id":r["round_id"],"teacher_beta":r["teacher_beta"],"successes":r["successes"],
            "episodes":r["completed_episodes"],"checkpoint_sha256":r["final_checkpoint"]["sha256"],
            "wandb":r["wandb"]} for r in rows]}


def complete_receipt(state):
    if (state.get("phase")!="COMPLETE" or [r["round_id"] for r in state.get("completed_rounds",[])]!=[1,2,3,4]
            or any(r["wandb"].get("online_metrics_verified") is not True for r in state["completed_rounds"])):
        raise RuntimeError("all training/success/W&B gates must complete")
    windows=state["config"]["execution"]["window_count"]
    return {"schema":"roborender.monitored-smoke-completion.v1",**summary(state),
        "optimizer_updates":4*windows,"learner_chunks":4*windows*12,"transitions":4*windows*972,
        "success_cases_per_round":state["config"]["validation"]["completed_episodes"],
        "qualification":"bounded_end_to_end_smoke_not_policy_convergence_or_all_failure_modes"}


def make_site(operations,root,state):
    from pipeline.deployment.resident_campaign_site import ResidentSite
    argv=operations["commands"]["prepare"]["argv"]
    def argument(flag):return argv[argv.index(flag)+1]
    request={"phase":"prepare","run_id":state["run_id"],"round_id":state["round_id"],
        "campaign_root":str(root),"config":state["config"],"infrastructure_attempts":state.get("infrastructure_attempts",{})}
    return ResidentSite(json.loads(Path(argument("--site")).read_bytes()),request,
        argument("--resident-pool"),argument("--resident-pool-sha256"),
        management_release=argument("--management-release") if "--management-release" in argv else None,
        management_release_sha256=argument("--management-release-sha256") if "--management-release-sha256" in argv else None)


def compute_status(operations,root,state):
    value=make_site(operations,root,state).allocation("inspect")
    replicas=value.get("replicas",[])
    assigned=value.get("present") is True and len(replicas)==1 and bool(replicas[0].get("node_id"))
    path=round_runtime_path(root,state["round_id"],state.get("infrastructure_attempts",{}))/"allocation-readiness.json"
    readiness=json.loads(path.read_bytes()) if path.exists() else {}
    return {"checked_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "request_present":value.get("present"),"node_assigned":assigned,
        "replicas":replicas,"allocation_phase":readiness.get("phase"),
        "allocation_phase_elapsed_seconds":readiness.get("phase_elapsed_seconds")}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--first-round-config",type=Path,required=True)
    parser.add_argument("--operations",type=Path,required=True)
    parser.add_argument("--monitor-root",type=Path,required=True)
    parser.add_argument("--interval-seconds",type=int,default=60)
    parser.add_argument("--max-boundary-attempts",type=int,default=12)
    parser.add_argument("--wall-budget-seconds",type=int,default=42600)
    parser.add_argument("--wandb-entity",default="nvidia")
    args=parser.parse_args()
    if not 30<=args.interval_seconds<=1800 or not 1<=args.max_boundary_attempts<=48 or not 60<=args.wall_budget_seconds<=42600:
        parser.error("bounded monitoring/retry budgets required")
    root=args.root.resolve(strict=True);output=args.monitor_root.resolve()
    if not output.is_relative_to(root.parent) or output==root:
        parser.error("monitor output must be a separate scoped campaign sibling")
    output.mkdir(parents=True,exist_ok=True)
    stop=[False]
    def interrupted(signum,frame):stop[0]=True
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    started=time.monotonic();deadline=started+args.wall_budget_seconds
    operations=json.loads(args.operations.read_bytes())
    current_ops=args.operations;attempt=0;child=None;log=None
    with (output/"manager.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def report(event,**extra):
            state=json.loads((root/"state.json").read_bytes())
            payload={"schema":"roborender.campaign-monitor.v1","event":event,**summary(state),
                "checked_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                "elapsed_seconds":time.monotonic()-started,"attempt":attempt,**extra}
            atomic(output/"status.json",payload)
            with (output/"events.jsonl").open("a") as stream:stream.write(json.dumps(payload,sort_keys=True)+"\n")
            print(json.dumps(payload,sort_keys=True),flush=True)
        try:
            while time.monotonic()<deadline and not stop[0]:
                state=json.loads((root/"state.json").read_bytes())
                if state["phase"]=="COMPLETE":
                    receipt=complete_receipt(state)
                    receipt["compute"]=compute_status(operations,root,state)
                    if receipt["compute"]["request_present"] is not False:
                        raise RuntimeError("final H200 release is not confirmed")
                    atomic(output/"completion.json",receipt);report("COMPLETE",compute=receipt["compute"])
                    return
                if child is not None and child.poll() is None:
                    try:compute=compute_status(operations,root,state)
                    except Exception as exc:compute={"health":"unknown","error_type":type(exc).__name__}
                    report("MONITOR",compute=compute)
                    until=min(deadline,time.monotonic()+args.interval_seconds)
                    while time.monotonic()<until and child.poll() is None and not stop[0]:time.sleep(1)
                    continue
                if log is not None:log.close();log=None
                if state["phase"]=="ABANDONED":
                    failure=json.loads((root/"failure.json").read_bytes())
                    if not retryable_boundary(state,failure) or attempt>=args.max_boundary_attempts:
                        raise RuntimeError("not a safe next-round startup failure; reconciliation required")
                    attempt+=1
                    rid=uuid.uuid4().hex
                    command=[sys.executable,"-m","pipeline.deployment.round_boundary_recovery",
                        "--root",str(root),"--first-round-config",str(args.first_round_config),
                        "--operations",str(args.operations),"--recovery-id",rid,"--admit"]
                    with (output/f"admit-{attempt:03d}.log").open("xb") as stream:
                        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=min(900,max(1,deadline-time.monotonic())))
                    if result.returncode:
                        report("ADMISSION_WAIT",admission_exit_code=result.returncode)
                        until=min(deadline,time.monotonic()+min(300,30*attempt))
                        while time.monotonic()<until and not stop[0]:time.sleep(1)
                        continue
                    current_ops=root/"recoveries"/rid/"operations.json"
                    report("BOUNDARY_RECOVERED",recovery_id=rid)
                elif child is not None:
                    raise RuntimeError("campaign exited without a safe abandoned boundary; no automatic replay")
                log=(output/f"campaign-{attempt:03d}.log").open("xb")
                command=[sys.executable,"-m","pipeline.dagger.beta_campaign","--root",str(root),
                    "--first-round-config",str(args.first_round_config),"--operations",str(current_ops),
                    "--wandb-entity",args.wandb_entity,"--execute"]
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                report("CAMPAIGN_STARTED",controller_pid=child.pid)
            raise TimeoutError("monitor wall deadline or stop request; scoped campaign cleanup required")
        except BaseException as exc:
            report("FAILED",error_type=type(exc).__name__)
            raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=650)
                except subprocess.TimeoutExpired:
                    # Do not blind-kill a native owner; the independent lease
                    # watchdog retains responsibility for scoped cleanup.
                    report("CONTROLLER_CLEANUP_UNCONFIRMED")
            if log is not None:log.close()


if __name__=="__main__":main()
