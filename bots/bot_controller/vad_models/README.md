# Vendored VAD model

`silero_vad.onnx` is the model bundled with **silero-vad 6.2.1** (MIT licence).

```
SHA-256  1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3
size     2327524 bytes
```

Verified byte-identical to `silero_vad/data/silero_vad.onnx` from the 6.2.1 wheel.

It is vendored rather than installed because the `silero-vad` package requires `torch` even on
the ONNX path (~200 MB of RSS per worker process, against a 1536Mi limit shared by four prefork
children). We drive the graph directly with `onnxruntime` instead — see `../silero_vad.py`.

The 512-sample window / 64-sample context / `(2, 1, 128)` state contract has been fixed since
v5.0, so anything written against `silero-vad >= 5.1` applies. v6 is materially better than v5
on noise (noise-only accuracy 0.61 → 0.87), which is the reason to be on it.

To verify after any change:

```bash
shasum -a 256 silero_vad.onnx
```
