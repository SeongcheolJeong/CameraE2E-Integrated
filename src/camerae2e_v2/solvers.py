"""Common local solver adapters for RayOptics, FDTD, and TCAD."""

from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from pyisetcam import (
    camerae2e_physics_simulation_commands,
    camerae2e_physics_simulation_manifest,
    camerae2e_physics_simulation_validate,
)

from .models import FidelityLevel, ReadinessTier
from .project import Project


@dataclass(frozen=True)
class SolverSpec:
    family: str
    stage_ids: tuple[str, ...]
    output_artifact_type: str
    readiness_tier: ReadinessTier
    truth_boundary: str


_SPECS = {
    "rayoptics": SolverSpec(
        family="rayoptics",
        stage_ids=("rayoptics_lens_psf",),
        output_artifact_type="rayoptics_solver_manifest",
        readiness_tier=ReadinessTier.PROXY,
        truth_boundary=(
            "RayOptics output is geometric ray-histogram evidence. It is not a diffraction "
            "or wave-optics sign-off result."
        ),
    ),
    "fdtd": SolverSpec(
        family="fdtd",
        stage_ids=("fdtd_optical_lut",),
        output_artifact_type="fdtd_solver_manifest",
        readiness_tier=ReadinessTier.PROXY,
        truth_boundary=(
            "FDTD output requires convergence, measured stack geometry, and material n/k "
            "evidence before quantitative camera-module use."
        ),
    ),
    "tcad": SolverSpec(
        family="tcad",
        stage_ids=("tcad_generation_map", "tcad_devsim_collection"),
        output_artifact_type="tcad_solver_manifest",
        readiness_tier=ReadinessTier.CALIBRATION_REQUIRED,
        truth_boundary=(
            "TCAD/DEVSIM collection output is calibration-required and must share lineage "
            "with its FDTD generation map."
        ),
    ),
}


class SolverAdapter:
    def __init__(self, family: str) -> None:
        if family not in _SPECS:
            raise KeyError(f"Unknown solver family: {family}")
        self.spec = _SPECS[family]

    def prepare(
        self,
        project: Project,
        *,
        candidate: dict[str, Any],
        job_id: str,
        dependencies: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        manifest = camerae2e_physics_simulation_manifest()
        stages = [
            stage
            for stage in manifest.get("stages", [])
            if stage.get("stage_id") in self.spec.stage_ids
        ]
        commands = []
        for stage_id in self.spec.stage_ids:
            commands.extend(
                camerae2e_physics_simulation_commands(
                    stages=stages,
                    stage_id=stage_id,
                    include_expensive=True,
                )
            )
        payload = {
            "schema_version": "camerae2e_solver_input_v2",
            "job_id": job_id,
            "family": self.spec.family,
            "candidate": candidate,
            "stages": stages,
            "commands": commands,
            "truth_boundary": self.spec.truth_boundary,
        }
        artifact = project.artifacts.put_json(
            payload,
            artifact_type=f"{self.spec.family}_solver_input",
            fidelity_level=FidelityLevel.SOLVER,
            readiness_tier=self.spec.readiness_tier,
            source="camerae2e_v2.solver.prepare",
            dependencies=dependencies,
            validation={"prepared": True, "stage_count": len(stages)},
        )
        return {"payload": payload, "artifact": artifact.model_dump(mode="json")}

    def submit(
        self,
        prepared: dict[str, Any],
        *,
        execute: bool,
        acknowledge_expensive: bool,
        timeout_s: int,
    ) -> dict[str, Any]:
        commands = list(prepared["payload"].get("commands", []))
        if not execute:
            return {
                "status": "existing_artifact_validation",
                "executed": False,
                "commands": commands,
                "note": "No solver command was launched; existing artifacts are validated.",
            }
        expensive = [item for item in commands if item.get("cost_tier") == "external_expensive"]
        if expensive and not acknowledge_expensive:
            raise ValueError(
                "Solver execution contains external_expensive commands; "
                "set acknowledge_expensive=true to run them"
            )
        selected = expensive or commands
        if not selected:
            raise ValueError(f"No executable command is registered for {self.spec.family}")
        executions = []
        started = time.monotonic()
        for command in selected:
            remaining = max(1, int(timeout_s - (time.monotonic() - started)))
            if remaining <= 1 and time.monotonic() - started >= timeout_s:
                raise TimeoutError(f"{self.spec.family} solver validation timed out")
            command_text = str(command.get("command", "")).strip()
            if not command_text:
                continue
            completed = subprocess.run(
                ["/bin/zsh", "-lc", command_text],
                capture_output=True,
                text=True,
                timeout=remaining,
                check=False,
            )
            executions.append(
                {
                    "name": command.get("command_id"),
                    "command": command_text,
                    "command_argv_preview": shlex.split(command_text)[:8],
                    "returncode": completed.returncode,
                    "stdout_tail": completed.stdout[-4000:],
                    "stderr_tail": completed.stderr[-4000:],
                }
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{self.spec.family} command failed ({completed.returncode}): "
                    f"{completed.stderr[-1000:]}"
                )
        return {"status": "completed", "executed": True, "executions": executions}

    def collect(
        self,
        project: Project,
        prepared: dict[str, Any],
        submission: dict[str, Any],
    ) -> dict[str, Any]:
        manifest = camerae2e_physics_simulation_manifest()
        selected_stages = [
            stage
            for stage in manifest.get("stages", [])
            if stage.get("stage_id") in self.spec.stage_ids
        ]
        validation = camerae2e_physics_simulation_validate(
            {"schema_version": manifest.get("schema_version"), "stages": selected_stages},
            strict=False,
        )
        consumable_outputs = [
            dict(output)
            for stage in selected_stages
            for output in stage.get("outputs", [])
            if output.get("exists") and output.get("kind") == "file"
        ]
        payload = {
            "schema_version": "camerae2e_solver_result_v2",
            "family": self.spec.family,
            "submission": submission,
            "stages": selected_stages,
            "validation": validation,
            "input_artifact_hash": prepared["artifact"]["hash"],
            "truth_boundary": self.spec.truth_boundary,
            "consumable_outputs": consumable_outputs,
        }
        artifact = project.artifacts.put_json(
            payload,
            artifact_type=self.spec.output_artifact_type,
            fidelity_level=FidelityLevel.SOLVER,
            readiness_tier=self.spec.readiness_tier,
            source="camerae2e_v2.solver.collect",
            dependencies=[prepared["artifact"]["hash"]],
            validation=validation,
        )
        return {
            "family": self.spec.family,
            "status": submission["status"],
            "validation": validation,
            "artifact": artifact.model_dump(mode="json"),
            "truth_boundary": self.spec.truth_boundary,
            "consumable_outputs": consumable_outputs,
        }


def solver_adapter(family: str) -> SolverAdapter:
    return SolverAdapter(family)


def candidate_escalation_plan(
    candidate: dict[str, Any],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    parameters = dict(candidate.get("parameters", candidate))
    paths = set(parameters)
    recommendations = []

    def add(family: str, reason: str, priority: int) -> None:
        spec = _SPECS[family]
        recommendations.append(
            {
                "family": family,
                "priority": priority,
                "reason": reason,
                "readiness_tier": spec.readiness_tier.value,
                "truth_boundary": spec.truth_boundary,
            }
        )

    if any(path.startswith("optics.") for path in paths):
        add("rayoptics", "Lens/PSF variables influence candidate ranking.", 1)
    if any(path.startswith("sensor.ocl") or path.startswith("fdtd.") for path in paths):
        add("fdtd", "OCL/CRA/crosstalk variables require field-resolved optical evidence.", 1)
    if any(
        path.startswith("sensor.pixel")
        or path.startswith("sensor.binning")
        or path.startswith("tcad.")
        for path in paths
    ):
        add("tcad", "Pixel/readout variables require carrier-collection sensitivity checks.", 2)
    if not recommendations:
        add("rayoptics", "Confirm the baseline lens field PSF for the selected module.", 3)
    recommendations.sort(key=lambda item: (item["priority"], item["family"]))
    return {
        "schema_version": "camerae2e_candidate_escalation_v2",
        "strict": strict,
        "candidate": candidate,
        "recommendations": recommendations,
        "execution_policy": (
            "Only shortlisted candidates are submitted to expensive local solvers. "
            "Solver outputs must be converted to immutable artifacts before reuse."
        ),
    }
