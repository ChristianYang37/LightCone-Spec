import math

from lightcone_spec.protocol import PAPER_NODES, default_row_counts, materialize, paper_plan
from lightcone_spec.runner import _request_count
from lightcone_spec.server import adaptation_payload, server_session_key


def test_registered_node_order_and_counts():
    assert PAPER_NODES == (
        "preflight", "E3a", "TTS-Cal", "E1", "E2-r0", "E2-r1", "E2-r2", "E2-r3",
        "E4-screen", "E4-local", "E4-profile", "E3b-pilot", "E3b-final", "E1a",
        "E5-pilot", "E5-final", "E6-pilot", "E6-final", "E0-tune", "E0-pilot", "E0-final",
    )
    assert default_row_counts() == {
        "preflight": 10, "E3a": 360, "TTS-Cal": 288, "E1": 68,
        "E2-r0": 3364, "E2-r1": 844, "E2-r2": 214, "E2-r3": 57,
        "E4-screen": 48, "E4-local": 96, "E4-profile": 3,
        "E3b-pilot": 1920, "E3b-final": 5760, "E1a": 116,
        "E5-pilot": 2064, "E5-final": 5400, "E6-pilot": 242, "E6-final": 720,
        "E0-tune": 25920, "E0-pilot": 6912, "E0-final": 20736,
    }
    assert len(paper_plan()) == 21


def test_readable_ids_and_dynamic_e0_grid():
    jobs = materialize("E0-tune", valid_e0=[("m", "DFLASH", "task")])
    assert len(jobs) == 108 + 239
    assert jobs[0].job_id.startswith("E0-tune__000000__")
    assert all(len(job.job_id) < 180 for job in jobs)
    candidates = jobs[108 : 108 + 236]
    assert {method: sum(job.method == method for job in candidates) for method in {
        "onlinespec_ogd", "onlinespec_opt", "onlinespec_ens"
    }} == {"onlinespec_ogd": 64, "onlinespec_opt": 64, "onlinespec_ens": 108}
    assert len({repr(sorted(job.parameters.items())) for job in candidates}) == 236


def test_two_gpu_exclusivity_and_real_interface_probes():
    assert {job.gpu_count for job in materialize("E5-pilot")} == {2}
    assert {job.method for job in materialize("E6-pilot")[:2]} == {"static"}
    assert {job.method for job in materialize("E0-tune", valid_e0=[])[:108]} == {
        "static"
    }


def test_registered_axes_reach_runtime_parameters():
    tts = materialize("TTS-Cal")[0]
    payload = adaptation_payload(tts)
    assert payload["weight_update_mode"] == "full"
    assert payload["optimizer"]["name"] == "adam"
    assert payload["optimizer"]["weight_decay"] == 0
    assert payload["optimizer"]["grad_clip"] == 0
    assert payload["teacher_row_policy"] == "latest_update_round_only"
    assert payload["loss_position_decay"] == math.exp(-1 / 7)

    rounds = [materialize(f"E2-r{index}")[0] for index in range(4)]
    assert [(job.load, job.context) for job in rounds] == [
        ("c2", 4096), ("c4", 8192), ("c8", 16384), ("c16", 40928)
    ]
    assert {job.parameters["regime"] for job in rounds} == {
        "short_input_long_generation"
    }
    screen = materialize("E4-screen")
    assert len(screen) == 48
    assert {job.parameters["stride"] for job in screen} == {1, 50}
    assert {job.parameters["traffic"] for job in screen} == {
        "pure_decode", "mixed_prefill_decode"
    }


def test_session_key_reuses_scalar_recipe_but_not_layout():
    first, second = materialize("TTS-Cal")[:2]
    assert server_session_key(first) == server_session_key(second)
    lora = materialize("E1")[4]
    full = materialize("E1")[5]
    assert server_session_key(lora) != server_session_key(full)


def test_final_tail_gate_has_registered_request_mass():
    final_rows = materialize("E5-final", final_blocks=12)
    saturation = [
        job for job in final_rows if job.load == "saturation_soak"
    ]
    assert len(saturation) == 5 * 2 * 12
    assert _request_count(_MinimalConfig(), saturation[0]) == 840
    assert 12 * 840 >= 10_000


class _MinimalServer:
    requests_per_cell = 16


class _MinimalConfig:
    server = _MinimalServer()
