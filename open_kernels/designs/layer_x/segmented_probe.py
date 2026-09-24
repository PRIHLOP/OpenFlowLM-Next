"""Actual dense segmented down worker/schedule, with optional partial snapshots.

Full FFN mode runs the production up/gate/act/down body and schedule. Both modes
keep actual main-core scratch and pool/activation offsets, without other layers.
"""
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, InOut, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import Tile

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from ironutil import Pipeline, include_dirs
import xcommon as X

FULL = os.environ.get('PROBE_FULL_FFN') == '1'
DIAGNOSTIC = not FULL and os.environ.get('PROBE_PARTIALS', '1') == '1'
L = X.R.layout
OUT = L.A_OUT2
ACT_BYTES = max(L.A_BYTES, OUT + X.HID * 4 * len(X.FFN.DOWN_SEGMENTS))
if not X.FFN.DOWN_SEGMENTS or X.Q8:
    raise ValueError('segmented probe requires a segmented all-Q4 dense recipe')


@iron.jit(aiecc_flags=['--alloc-scheme=basic-sequential'])
def segmented(pool: In, act: InOut, *, source_hash: CompileTime[int] = 0):
    t = X.types()
    inc = include_dirs() + [str(HERE), str(HERE.parent / 'gemv_q4')]
    kernels = X.kernels(inc, t)
    wf = [ObjectFifo(t['elem'], name=f'w{c}', depth=2) for c in range(X.N_CORES)]
    xf = ObjectFifo(t['x'], name='x', depth=2)
    yf = [ObjectFifo(t['y'], name=f'y{c}', depth=2) for c in range(X.N_CORES)]

    def core_body(win, xin, yout, *args):
        buffers, functions = X.unpack_args(args)
        if FULL:
            X.ffn_body(win, xin, yout, buffers, functions)
        else:
            X.segmented_down_body(win, xin, yout, buffers, functions, DIAGNOSTIC)

    workers = [Worker(core_body,
                      fn_args=[wf[c].cons(), xf.cons(), yf[c].prod(),
                               *X.worker_args(X.core_buffers(t, c), kernels)],
                      tile=Tile(c, 2), stack_size=0x1800) for c in range(X.N_CORES)]
    wty = np.ndarray[(X.POOL_BYTES,), np.dtype[np.uint8]]
    aty = np.ndarray[(ACT_BYTES,), np.dtype[np.uint8]]

    def sequence(a_w, a_act, w_prods, x_prod, y_conss):
        pw, px, py = Pipeline(3), Pipeline(3), Pipeline(3)
        if FULL:
            X.ffn_sequence(pw, px, py, a_w, a_act, w_prods, x_prod, y_conss,
                           ACT_BYTES, L.A_XM, L.A_H, OUT,
                           L.POOL_FFN_UP, L.POOL_FFN_GATE, L.POOL_FFN_DOWN)
        else:
            X.segmented_down_sequence(pw, px, py, a_w, a_act, w_prods, x_prod, y_conss,
                                      ACT_BYTES, L.A_H, OUT, L.POOL_FFN_DOWN, DIAGNOSTIC)
        pw.finish()
        px.finish()
        py.finish()

    rt = Runtime(sequence, [wty, aty,
                            [f.prod(tile=Tile(c, 0)) for c, f in enumerate(wf)],
                            xf.prod(tile=Tile(1, 0)),
                            [f.cons(tile=Tile(c, 0)) for c, f in enumerate(yf)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = segmented
_sources = [Path(__file__), HERE / 'xcommon.py', HERE / 'gen_kernels.py', ROOT / 'ironutil.py',
            *sorted(HERE.glob('*.cc')), *sorted(HERE.glob('*.h')),
            *sorted((HERE.parent / 'gemv_q4').glob('*.h'))]
SPECIALIZE = {'source_hash': int(hashlib.sha256(b''.join(p.read_bytes() for p in _sources)
                         + b''.join(X.source_hash_inputs())
                         + repr((FULL, DIAGNOSTIC, X.C, X.FFN, L)).encode()).hexdigest()[:8], 16)}
