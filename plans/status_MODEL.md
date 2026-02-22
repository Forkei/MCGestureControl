# Status: MODEL

## Completed
- [x] Added `ControlTransformerV2` class to `control_model.py`
- [x] Input projection: Linear(671 -> 256)
- [x] Learnable positional encoding: max 30 positions
- [x] Transformer encoder: 6 layers, 8 heads, d_ff=512, dropout=0.3, batch_first=True
- [x] Exponential temporal pooling with learnable decay parameter (init -2.0)
- [x] 5 output heads (all Linear(256->128)->ReLU->Dropout(0.3)->Linear(128->N)):
  - Action head: 12 outputs, raw logits
  - Look head: 2 outputs, tanh [-1,1]
  - Hotbar head: 9 outputs, raw logits
  - Cursor head: 2 outputs, sigmoid [0,1]
  - InvClick head: 3 outputs, raw logits
- [x] Forward returns dict with keys: action_logits, look, hotbar_logits, cursor, inv_click_logits
- [x] `config_from_dict(d)` static method — builds from Contract 7 JSON config
- [x] `param_count()` method
- [x] Verified 3,510,429 parameters (~3.51M) — on target
- [x] Forward pass verified with padding masks
- [x] V1 `ControlTransformer` class preserved unchanged

## Interface Changes
- None — fully conforms to Contract 4 and Contract 7

## Notes
- The `temporal_decay` parameter is a single learnable scalar (init -2.0). With decay < 0, recent frames (end of sequence) get exponentially higher weight, same concept as v1 but now learnable.
- Padding mask is properly converted from (1.0=real, 0.0=pad) to PyTorch's convention (True=ignore) for `src_key_padding_mask`.
- The PyTorch nested tensor warning during forward pass is cosmetic and harmless.
