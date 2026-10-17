"""Profile MobileNetV4-S/M with ReLU / hard-swish / SiLU on a cloud Galaxy S24 (Qualcomm AI Hub).
Random-init weights (latency does not depend on weight values in FP mode). TFLite runtime, as in the AI Hub tutorial."""
import json, sys, time
import torch, timm
import qai_hub as hub

out = sys.argv[1]
device = hub.Device("Samsung Galaxy S24 (Family)")
shape = (1, 3, 224, 224)
jobs = {}
for size in ("small", "medium"):
    for act in ("relu", "hard_swish", "silu"):
        torch.manual_seed(0)
        kw = {} if act == "relu" else {"act_layer": act}
        m = timm.create_model(f"mobilenetv4_conv_{size}", pretrained=False, num_classes=1000, **kw).eval()
        kinds = sorted({type(x).__name__ for x in m.modules() if type(x).__name__ in ("ReLU", "Hardswish", "SiLU")})
        with torch.no_grad():
            ex = torch.export.export(m, (torch.rand(shape),))
        cj = hub.submit_compile_job(model=ex, device=device, input_specs=dict(image=shape),
                                    options="--target_runtime tflite", name=f"mnv4{size[0]}-{act}")
        jobs[f"{size}-{act}"] = {"activations": kinds, "compile": cj.job_id}
        print(size, act, kinds, cj.job_id, flush=True)
for key, j in jobs.items():
    cj = hub.get_job(j["compile"]); tm = cj.get_target_model()
    if tm is None:
        j["error"] = "compile failed"; continue
    pj = hub.submit_profile_job(model=tm, device=device, name=f"profile-{key}")
    j["profile"] = pj.job_id; print("profile", key, pj.job_id, flush=True)
for key, j in jobs.items():
    if "profile" not in j: continue
    prof = hub.get_job(j["profile"]).download_profile()
    s = prof["execution_summary"]
    det = prof.get("execution_detail", [])
    units = {}
    for layer in det:
        u = layer.get("compute_unit", "?"); units[u] = units.get(u, 0) + 1
    j.update(min_inference_ms=s["estimated_inference_time"] / 1000.0,
             peak_memory_mb=s.get("estimated_inference_peak_memory", 0) / 2**20,
             layers_by_compute_unit=units)
    print(key, j["min_inference_ms"], units, flush=True)
json.dump(jobs, open(out, "w"), indent=1)
print("DONE")
