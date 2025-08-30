#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust QAT (FX-like) script for YOLOv8-seg — fixed prepare_qat signature handling.
Edit CONFIG at top then run: python Quantize_aware_training.py
"""
import os
import re
import sys
import inspect
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# help FX by wrapping builtins that often break tracing
import torch.fx
for _fn in ("len", "list", "tuple", "range", "zip", "map"):
    try:
        torch.fx.wrap(_fn)
    except Exception:
        pass

# detect available QAT API (fx-style or new)
_prepare_qat_fn = None
_convert_fn = None
QConfigMapping = None
get_default_qat_qconfig = None
API_MODE = None

try:
    # fx API (older)
    from torch.ao.quantization.fx import prepare_qat_fx as _prepare_qat_fx
    from torch.ao.quantization.fx import convert_fx as _convert_fx
    from torch.ao.quantization.fx import QConfigMapping as QConfigMapping_fx
    from torch.ao.quantization import get_default_qat_qconfig as _gdq_fx
    _prepare_qat_fn = _prepare_qat_fx
    _convert_fn = _convert_fx
    QConfigMapping = QConfigMapping_fx
    get_default_qat_qconfig = _gdq_fx
    API_MODE = "fx"
except Exception:
    try:
        # new API
        from torch.ao.quantization import prepare_qat as _prepare_qat_new
        from torch.ao.quantization import convert as _convert_new
        from torch.ao.quantization.qconfig_mapping import QConfigMapping as QConfigMapping_new
        from torch.ao.quantization import get_default_qat_qconfig as _gdq_new
        _prepare_qat_fn = _prepare_qat_new
        _convert_fn = _convert_new
        QConfigMapping = QConfigMapping_new
        get_default_qat_qconfig = _gdq_new
        API_MODE = "new"
    except Exception:
        API_MODE = None

if API_MODE is None:
    raise ImportError("No suitable PyTorch QAT API found (need prepare_qat_fx/prepare_qat). Check torch version.")

# ------------------------- CONFIG -------------------------
MODEL_PATH = r"runs/segment/yolov8_custom_train5/weights/best.pt"
DATA_ROOT = r"D:/new_dataset1"
BACKEND = "fbgemm"        # "fbgemm" (x86) or "qnnpack" (ARM/Jetson)
IMGSZ = 640
EPOCHS = 12
BATCH = 16
LR0 = 1e-3
DEVICE = "0"
WORKERS = 4
EXCLUDE_UPSAMPLE = True
EXCLUDE_REGEX = None      # e.g. r"seg|mask|p3"
EXPORT_ONNX = True
CONVERT_INT8_PTH = True
OUTDIR = "qat_outputs_fixed_v3"
# ----------------------------------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def patch_inplace_activations(m: nn.Module):
    for mod in m.modules():
        if isinstance(mod, (nn.SiLU, nn.ReLU)):
            try:
                mod.inplace = False
            except Exception:
                pass

def write_data_yaml(path_yaml: Path, data_root: Path, names):
    data_root = data_root.resolve()
    lines = [
        f"path: {data_root.as_posix()}",
        f"train: { (data_root/'images'/'train').as_posix() }",
        f"val:   { (data_root/'images'/'val').as_posix() }",
    ]
    if isinstance(names, dict):
        names_list = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    elif isinstance(names, (list, tuple)):
        names_list = list(names)
    else:
        names_list = [f"class{i}" for i in range(80)]
    lines.append(f"names: {names_list}")
    path_yaml.write_text("\n".join(lines), encoding="utf-8")
    return path_yaml

def get_core_nnmodule(yolo_obj):
    cand = getattr(yolo_obj, "model", None)
    if cand is None:
        cand = yolo_obj
    if hasattr(cand, "model") and isinstance(cand.model, nn.Module):
        core = cand.model
    elif isinstance(cand, nn.Module):
        core = cand
    else:
        found = None
        try:
            for name, m in cand.named_children():
                if isinstance(m, nn.Module):
                    found = m
                    break
        except Exception:
            found = None
        if found is None:
            print("[ERROR] Could not discover a core nn.Module. Dumping cand:")
            print("----- BEGIN MODEL STRUCTURE -----")
            try:
                print(cand)
            except Exception as e:
                print("[failed to print model]:", e)
            print("-----  END MODEL STRUCTURE  -----")
            raise RuntimeError("Cannot locate core nn.Module for FX tracing.")
        core = found
    if not isinstance(core, nn.Module):
        raise RuntimeError(f"Discovered core is not nn.Module: {type(core)}")
    return core

def build_qconfig_mapping_fx_like(backend: str, exclude_upsample: bool, exclude_regex: str, core: nn.Module):
    qcfg = get_default_qat_qconfig(backend)
    mapping = QConfigMapping().set_global(qcfg)
    if exclude_upsample:
        mapping = mapping.set_object_type(nn.Upsample, None)
    if exclude_regex and core is not None:
        pattern = re.compile(exclude_regex, re.IGNORECASE)
        for name, _ in core.named_modules():
            if pattern.search(name):
                mapping = mapping.set_module_name(name, None)
    return mapping

def apply_qconfig_new_api(backend: str, exclude_upsample: bool, exclude_regex: str, core: nn.Module):
    qcfg = get_default_qat_qconfig(backend)
    core.qconfig = qcfg
    if exclude_upsample:
        for nm, module in core.named_modules():
            if isinstance(module, nn.Upsample):
                module.qconfig = None
    if exclude_regex:
        pattern = re.compile(exclude_regex, re.IGNORECASE)
        for nm, module in core.named_modules():
            if pattern.search(nm):
                module.qconfig = None
    return qcfg

def call_prepare_qat_dynamic(prepare_fn, model_module, qconfig_mapping_obj, example_inputs):
    """
    Call prepare_qat / prepare_qat_fx robustly across different torch versions
    by inspecting function signature and passing only supported args.
    - prepare_fn: function object (could be prepare_qat_fx or prepare_qat)
    - model_module: core nn.Module (must already be in train() if required)
    - qconfig_mapping_obj: QConfigMapping instance or None (for new API)
    - example_inputs: tuple or None
    """
    sig = inspect.signature(prepare_fn)
    params = sig.parameters
    kwargs = {}
    # If function accepts a 'qconfig_mapping' or 'qconfig' param, pass mapping
    if "qconfig_mapping" in params:
        kwargs["qconfig_mapping"] = qconfig_mapping_obj
    if "qconfig" in params:
        # some versions expect qconfig mapping under name 'qconfig'
        kwargs["qconfig"] = qconfig_mapping_obj
    # example inputs variants
    if "example_inputs" in params:
        kwargs["example_inputs"] = example_inputs
    # older FX API sometimes expects example_inputs as positional second arg.
    try:
        if len(kwargs) == 0:
            # call with just model
            return prepare_fn(model_module)
        else:
            return prepare_fn(model_module, **kwargs)
    except TypeError:
        # try fallback: positional (model, qconfig_mapping, example_inputs)
        try:
            if qconfig_mapping_obj is not None and example_inputs is not None:
                return prepare_fn(model_module, qconfig_mapping_obj, example_inputs)
            elif example_inputs is not None:
                return prepare_fn(model_module, example_inputs)
            else:
                return prepare_fn(model_module)
        except Exception as e:
            raise

def main():
    outdir = ensure_dir(Path(OUTDIR))
    data_root = Path(DATA_ROOT)

    # backend selection & set
    supported = getattr(torch.backends.quantized, "supported_engines", None)
    supported = list(supported) if supported is not None else ["fbgemm", "qnnpack"]
    chosen_backend = BACKEND if BACKEND in supported else ("fbgemm" if "fbgemm" in supported else supported[0])
    if chosen_backend != BACKEND:
        print(f"[WARN] Requested backend {BACKEND} not supported here. Falling back to {chosen_backend}.")
    torch.backends.quantized.engine = chosen_backend
    print(f"[INFO] Using quant backend: {torch.backends.quantized.engine}  (API mode: {API_MODE})")

    # load model
    print(f"[INFO] Loading YOLO model from: {MODEL_PATH}")
    yolo = YOLO(MODEL_PATH)
    core = get_core_nnmodule(yolo)
    print(f"[INFO] Core module type: {type(core)}")
    try:
        print("[DEBUG] core children:", [(n, type(m)) for n, m in core.named_children()])
    except Exception:
        pass

    # write data yaml
    names = getattr(core, "names", None) or getattr(yolo, "names", None)
    data_yaml = outdir / "data_qat.yaml"
    write_data_yaml(data_yaml, data_root, names)
    print(f"[INFO] Data yaml created at: {data_yaml}")

    # patch activations
    patch_inplace_activations(core)

    # prepare QAT
    example_inputs = (torch.randn(1, 3, IMGSZ, IMGSZ),)
    print("[INFO] Preparing model for QAT (must be in train mode before prepare)...")

    model_qat_core = None
    if API_MODE == "fx":
        qconfig_mapping = build_qconfig_mapping_fx_like(torch.backends.quantized.engine, EXCLUDE_UPSAMPLE, EXCLUDE_REGEX, core)
        core.train()
        try:
            model_qat_core = call_prepare_qat_dynamic(_prepare_qat_fn, core, qconfig_mapping, example_inputs)
        except Exception as e:
            print("[ERROR] prepare_qat_fx failed with exception:")
            print(e)
            print("If TraceError, inspect printed core structure above; may need monkey-patch of blocks.")
            sys.exit(1)
    else:
        # new API: set qconfig attributes on modules
        apply_qconfig_new_api(torch.backends.quantized.engine, EXCLUDE_UPSAMPLE, EXCLUDE_REGEX, core)
        core.train()
        try:
            # call prepare dynamically without passing a mistaken object as 'qconfig_mapping'
            model_qat_core = call_prepare_qat_dynamic(_prepare_qat_fn, core, None, example_inputs)
        except Exception as e:
            print("[ERROR] prepare_qat (new API) failed with exception:")
            print(e)
            print("If TraceError, inspect printed core structure above; may need monkey-patch of blocks.")
            sys.exit(1)

    # attach back into YOLO wrapper
    if hasattr(yolo, "model") and hasattr(yolo.model, "model"):
        yolo.model.model = model_qat_core
        print("[INFO] Attached QAT core to yolo.model.model")
    elif hasattr(yolo, "model") and (yolo.model is core):
        yolo.model = model_qat_core
        print("[INFO] Replaced yolo.model with QAT core")
    else:
        try:
            setattr(yolo, "model", model_qat_core)
            print("[INFO] Attached QAT core to yolo.model (fallback)")
        except Exception:
            print("[WARN] Could not attach QAT core automatically; proceeding with model_qat_core standalone")
            yolo.model = model_qat_core

    # fine-tune with ultralytics trainer
    print("[INFO] Starting QAT fine-tune (Ultralytics trainer)...")
    yolo.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        lr0=LR0,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE
    )

    # retrieve trained core
    trained_core = None
    try:
        if hasattr(yolo, "model") and hasattr(yolo.model, "model"):
            trained_core = yolo.model.model
        elif hasattr(yolo, "model"):
            trained_core = yolo.model
    except Exception:
        trained_core = model_qat_core
    model_qat_trained = trained_core

    # eval + disable observers + freeze bn stats
    model_qat_trained.eval()
    for m in model_qat_trained.modules():
        if hasattr(m, "disable_observer"):
            try:
                m.disable_observer()
            except Exception:
                pass
        if hasattr(m, "freeze_bn_stats"):
            try:
                m.freeze_bn_stats()
            except Exception:
                pass

    # convert to quantized model
    print("[INFO] Converting QAT-prepared graph to quantized model...")
    model_int8 = None
    try:
        model_int8 = _convert_fn(model_qat_trained)
    except Exception as e:
        print("[WARN] convert failed:", e)
        model_int8 = None

    if model_int8 is not None and CONVERT_INT8_PTH:
        pth_path = outdir / "model_qat_int8.pth"
        torch.save(model_int8.state_dict(), pth_path.as_posix())
        print(f"[INFO] Saved PyTorch INT8 state_dict to: {pth_path}")

    # export ONNX (Q/DQ)
    if EXPORT_ONNX:
        onnx_path = outdir / "model_qat_qdq.onnx"
        print(f"[INFO] Exporting ONNX with Q/DQ to: {onnx_path}")
        dummy = torch.randn(1, 3, IMGSZ, IMGSZ)
        try:
            torch.onnx.export(
                model_qat_trained, dummy, onnx_path.as_posix(),
                opset_version=13,
                input_names=["images"],
                output_names=["output"],
                dynamic_axes={"images": {0: "batch"}}
            )
            print("[INFO] ONNX exported:", onnx_path)
        except Exception as e:
            print("[WARN] ONNX export failed:", e)

    print("\n[✅ DONE] QAT flow finished. Outputs in:", outdir)

if __name__ == "__main__":
    main()

