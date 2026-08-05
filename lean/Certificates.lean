/-!
# The Ternary Packing Ladder — kernel certificates

Standalone: no imports. Every theorem is decided by the Lean 4 kernel on
exact natural-number arithmetic. The ladder rows state that Q trits fit in
P bits (3^Q < 2^P) and that P is minimal (2^(P-1) < 3^Q) — the tightness
that makes these the only record block sizes.
-/

set_option maxRecDepth 1000000
set_option exponentiation.threshold 130000

-- Rung 1: 5 trits per byte (the industry format's core)
theorem rung1_fits  : (3:Nat) ^ 5 < 2 ^ 8 := by decide
theorem rung1_tight : (2:Nat) ^ 7 < 3 ^ 5 := by decide

-- Rung 2: 41 trits in 65 bits
theorem rung2_fits  : (3:Nat) ^ 41 < 2 ^ 65 := by decide
theorem rung2_tight : (2:Nat) ^ 64 < 3 ^ 41 := by decide

-- Rung 3: 306 trits in 485 bits
theorem rung3_fits  : (3:Nat) ^ 306 < 2 ^ 485 := by decide
theorem rung3_tight : (2:Nat) ^ 484 < 3 ^ 306 := by decide

-- Rung 4: 15601 trits in 24727 bits
theorem rung4_fits  : (3:Nat) ^ 15601 < 2 ^ 24727 := by decide
theorem rung4_tight : (2:Nat) ^ 24726 < 3 ^ 15601 := by decide

-- Law B's zero-tax identity, certified at the bit level:
-- 665 = 306 + 306 + 53 assembles at exactly 1055 = 485 + 485 + 85 bits
theorem law_b_665_fits  : (3:Nat) ^ 665 < 2 ^ 1055 := by decide
theorem law_b_665_tight : (2:Nat) ^ 1054 < 3 ^ 665 := by decide
theorem law_b_53_fits   : (3:Nat) ^ 53 < 2 ^ 85 := by decide
theorem law_b_53_tight  : (2:Nat) ^ 84 < 3 ^ 53 := by decide
