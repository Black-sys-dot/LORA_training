"""Torch 2.6 infer_schema rejects PEP 585 list[Tensor]; ai-toolkit HEAD uses it."""

from __future__ import annotations


def apply() -> None:
    try:
        import typing

        import torch
        from torch._library import infer_schema as m

        tensor = torch.Tensor
        m.SUPPORTED_RETURN_TYPES[list[tensor]] = "Tensor[]"
        m.SUPPORTED_RETURN_TYPES[typing.List[tensor]] = "Tensor[]"

        extra = {}
        for typ, schema in list(m.SUPPORTED_PARAM_TYPES.items()):
            origin = typing.get_origin(typ)
            args = typing.get_args(typ)
            if origin in (list, typing.List, typing.Sequence) and args:
                try:
                    extra[list[args[0]]] = schema
                except TypeError:
                    pass
        m.SUPPORTED_PARAM_TYPES.update(extra)
        print("schema_compat: list[Tensor] registered for infer_schema", flush=True)
    except Exception as e:
        print(f"schema_compat apply skipped: {type(e).__name__}: {e}", flush=True)


def patch_toolkit_files(root) -> int:
    import re
    from pathlib import Path

    n = 0
    root = Path(root)
    
    # 1. Patch type annotations for PyTorch 2.6 schema compatibility
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "list[torch.Tensor]" in text or "-> list[Tensor]" in text:
            orig = text
            if not re.search(r"^import typing\b", text, re.M):
                lines = text.splitlines(True)
                insert_at = 0
                for i, line in enumerate(lines):
                    if line.startswith("from __future__"):
                        insert_at = i + 1
                lines.insert(insert_at, "import typing\n")
                text = "".join(lines)
            text = text.replace("list[torch.Tensor]", "typing.List[torch.Tensor]")
            text = text.replace("-> list[Tensor]", "-> typing.List[Tensor]")
            if text != orig:
                path.write_text(text, encoding="utf-8")
                n += 1
                print(f"patched annotations {path.name}", flush=True)

    # 2. Patch BaseSDTrainProcess.py to ignore SageMaker .sagemaker-uploaded marker files
    train_proc = root / "jobs" / "process" / "BaseSDTrainProcess.py"
    if train_proc.exists():
        t_text = train_proc.read_text(encoding="utf-8")
        old_filter = "paths = [p for p in paths if os.path.exists(p)]"
        new_filter = "paths = [p for p in paths if os.path.exists(p) and not p.endswith('.sagemaker-uploaded') and not p.endswith('.yaml') and not p.endswith('.tmp') and p.endswith('.safetensors')]"
        if old_filter in t_text:
            t_text = t_text.replace(old_filter, new_filter)
            print("patched BaseSDTrainProcess.py paths filter to ignore .sagemaker-uploaded markers", flush=True)
            n += 1

        old_clean = "safetensors_files = [f for f in items if f.endswith('.safetensors')]"
        new_clean = "safetensors_files = [f for f in items if f.endswith('.safetensors') and not f.endswith('.sagemaker-uploaded')]"
        if old_clean in t_text:
            t_text = t_text.replace(old_clean, new_clean)
            print("patched BaseSDTrainProcess.py clean_up_saves to ignore .sagemaker-uploaded markers", flush=True)
            n += 1
        train_proc.write_text(t_text, encoding="utf-8")

    # 3. Patch network_mixins.py to ensure safetensors loader is used even if path has marker
    mixins = root / "toolkit" / "network_mixins.py"
    if mixins.exists():
        m_text = mixins.read_text(encoding="utf-8")
        old_load = 'if os.path.splitext(file)[1] == ".safetensors":'
        new_load = 'if str(file).endswith(".safetensors") or ".safetensors" in str(file):'
        if old_load in m_text:
            m_text = m_text.replace(old_load, new_load)
            print("patched network_mixins.py to safely load safetensors", flush=True)
            n += 1
        mixins.write_text(m_text, encoding="utf-8")

    return n
