# Doom-in-a-Transformer: Research & Feasibility

## The Goal

Compile Doom's game logic directly into transformer weights using ModelScriptor, so that each forward pass of the transformer computes one game tick deterministically. This is fundamentally different from models that *hallucinate* game frames — we would be *computing exact game state*.

---

## Prior Art: What Exists Today

### Category 1: Models That Hallucinate Game Frames (Learned Video Prediction)

These models learn to generate frames that *look like* games but don't compute game logic. They are video prediction models.

**GameNGen** (Google Research + Tel Aviv University, August 2024)
- Paper: [arxiv.org/abs/2408.14837](https://arxiv.org/abs/2408.14837)
- Modified Stable Diffusion v1.4, trained on 900M frames of Doom gameplay
- Runs at 20 FPS on a single TPU
- Two-phase training: (1) RL agent plays Doom and sessions are recorded, (2) diffusion model predicts next frame conditioned on past frames + actions
- PSNR of 29.4 (comparable to JPEG compression)
- **Critical limitation:** Only ~3 seconds of memory. Approximates game state (health, ammo, doors) but drifts over time. Hallucinating plausible frames, not computing exact game state.
- **Reception:** Major media coverage (VentureBeat, PC Gamer, Tom's Hardware, Hacker News front page). Impressed as a tech demo but critics correctly noted it's "a game engine only in the most generous interpretation."
- [gamengen.github.io](https://gamengen.github.io/)

**DIAMOND** (NeurIPS 2024 Spotlight)
- "Diffusion As a Model Of eNvironment Dreams" — [diamond-wm.github.io](https://diamond-wm.github.io/)
- Diffusion world model for RL, focused on Atari games and CS:GO
- Same fundamental approach: learned video prediction, not deterministic computation

**Oasis by Decart** (November 2024)
- Transformer-based (not diffusion), generates Minecraft-like gameplay at 20 FPS
- Autoregressive frame generation conditioned on user keyboard/mouse input
- Uses DiT backbone + ViT autoencoder
- Same category: learned approximation, not compiled game logic
- [oasis-model.github.io](https://oasis-model.github.io/)

**Google Genie 2** (December 2024)
- Foundation world model, generates interactive 3D environments from a single image
- Can maintain consistent worlds for up to ~60 seconds

### Category 2: RL Agents That Play Doom (Decision-Making)

**VizDoom** (2016-present)
- Standard RL benchmark platform built on Doom
- Neural network is a *policy network* — looks at pixels and decides which buttons to press
- The actual game runs on the original Doom engine
- Completely different from what we're doing. Old news since 2016.
- [github.com/Farama-Foundation/ViZDoom](https://github.com/Farama-Foundation/ViZDoom)

### Category 3: Compiled/Programmatic Transformer Computation (Our Category)

**Tracr** (Google DeepMind, January 2023)
- "TRAnsformer Compiler for RASP" — [arxiv.org/abs/2301.05062](https://arxiv.org/abs/2301.05062)
- Compiles RASP programs into transformer weights
- Example programs: sorting, counting tokens, balanced parentheses
- RASP is a restricted language, not designed for anything close to game complexity
- **Archived April 2025.** Built on JAX/Haiku (also being sunset).
- [github.com/google-deepmind/tracr](https://github.com/google-deepmind/tracr)

**RASP** (Weiss et al., ICML 2021)
- "Thinking Like Transformers" — [arxiv.org/abs/2106.06981](https://arxiv.org/abs/2106.06981)
- Programming language mapping to transformer primitives (attention + feedforward)
- Theoretical foundation that Tracr builds on

**Percepta transformer-vm** (March 2025) — **CLOSEST COMPETITOR**
- Compiles C → WebAssembly → transformer weights
- Standard softmax-ReGLU transformer, weights computed analytically (not trained)
- Deterministic execution, 100% accuracy
- Demos: Sudoku solver, Collatz conjecture, multi-digit addition
- Uses "First Futamura projection" — specific programs baked into FFN weights
- Karpathy praised it publicly
- **Has NOT been used for Doom or anything graphical**
- Reception was "fascinating" but also "but why?"
- [github.com/Percepta-Core/transformer-vm](https://github.com/Percepta-Core/transformer-vm)

**Looped Transformers as Programmable Computers** (Giannou et al., ICML 2023)
- [arxiv.org/abs/2301.13196](https://arxiv.org/abs/2301.13196)
- Proves 13-layer looped transformer can simulate a small instruction-set computer
- Constructive weight matrices (not learned)
- Theoretical — demonstrates Turing-completeness but not practical software

**Cajal** (Velez-Ginorio, November 2025)
- [arxiv.org/abs/2511.13769](https://arxiv.org/abs/2511.13769)
- Typed, higher-order language that compiles to "linear neurons"
- Focus: making compiled neural components differentiable for co-training

**ALTA** (Shaw et al., TMLR 2025)
- [arxiv.org/abs/2410.18077](https://arxiv.org/abs/2410.18077)
- Extends RASP/Tracr with loops and Universal Transformer compilation
- Academic/research-focused

---

## Key Insight: Why This Hasn't Been Done

Nobody has compiled a graphical game into transformer weights because:

1. **Scale gap:** Existing compilers handle programs with ~10-100 operations. A game requires thousands.
2. **Rendering:** No one has attempted raycasting or frame buffer output in compiled weights.
3. **Statefulness:** Games maintain state between frames, requiring input/output state management.
4. **Trig functions:** Raycasting requires sin/cos, which need FFN-based lookup table approximation.
5. **Iteration:** Raycasting loops must be unrolled into transformer layers.

The gap between "Sudoku in transformer weights" and "Doom in transformer weights" is enormous. That's exactly why doing it would matter.

---

## Novelty Claim

| Dimension | GameNGen | Percepta | ModelScriptor Doom |
|-----------|----------|----------|--------------------|
| Deterministic? | No — drifts after ~3s | Yes | Yes |
| Actual game logic? | No — hallucinated frames | Yes (small programs) | Yes |
| Graphical output? | Yes (learned) | No | Yes (computed) |
| Scale | Huge model, trained | Small programs | Game-scale logic |
| Approach | Trained on 900M frames | C → WASM → weights | Computation graph → weights |

**Our claim:** First deterministic, exact game running as compiled transformer weights with graphical output. Not hallucinated. Not a policy network. Not a video predictor. Actual game logic in actual weights.

**Differentiation from GameNGen:** "GameNGen hallucinates frames that look like Doom. We compute exact game state. That's like comparing a Magic 8-Ball to a CPU."

**Differentiation from Percepta:** "Percepta compiles Sudoku solvers. We compile Doom." Also architecturally different — native transformer operations vs. WASM interpreter encoding.

---

## Technical Approach: Minimal Viable Doom

### What "Doom" Needs to Mean

Not full Doom. A recognizable first-person Doom-like experience:
- First-person corridor renderer (raycasting)
- WASD movement + turning
- A simple grid-based map
- Doom-like wall textures (or solid colors with Doom aesthetic)
- Possibly: one enemy, shooting
- Tiny resolution: 32x32 or 64x64
- Each forward pass = one game tick

This follows the proud tradition of "Can it run Doom?" stunts, which are always simplified.

### Core Technical Challenges

**1. Raycasting**
- For each screen column, cast a ray from the player position at an angle
- Step along the ray until hitting a wall (DDA algorithm)
- Compute wall height from distance (perspective projection)
- This is fundamentally: loop (step ray) → check grid cell → compute distance → compute column height
- Loops become unrolled transformer layers
- DDA stepping is arithmetic (addition, comparison) — well within ModelScriptor's capabilities

**2. Trigonometry (sin/cos)**
- Raycasting needs sin/cos for ray direction calculation
- ModelScriptor's `map_to_table` can implement lookup tables in FFN layers
- Discretize angles (e.g., 256 angles) and store sin/cos as lookup entries
- Alternatively: CORDIC-style iterative approximation (fits nicely into layered computation)

**3. State Management**
- Game state (player x, y, angle, health, etc.) encoded as input tokens
- Transformer computes one tick: new state + frame buffer
- Output state fed back as input for next tick
- State is a small fixed-size vector — straightforward to encode

**4. Frame Buffer Output**
- Output is a flattened pixel array (32x32 = 1024 values, or 64x64 = 4096)
- Each pixel is a color index or RGB value
- Could output as sequence of tokens or as a single large vector

**5. Scale Estimation**
- 32 screen columns × ~16 max ray steps = ~512 ray-step operations
- Each ray step: ~5-10 basic operations (position update, grid check, distance calc)
- Total: ~2,500-5,000 operations per frame
- Plus state update, input handling, rendering: maybe 10,000 operations total
- This translates to a transformer with potentially hundreds of layers
- Feature assignment optimization (ModelScriptor's constraint solver) is critical here

### Possible Simplifications
- Fixed camera height (no look up/down) — classic Doom didn't have this either
- Grid-based map (no BSP tree needed)
- No floor/ceiling textures (solid colors)
- No sprites initially (add enemy as a billboard later)
- No lighting (or simple distance-based dimming)
- Very low resolution
- Quantized angles and positions

---

## Viral Strategy

### The Headline
"I Made a Transformer Run Doom — Not by Training It, but by Compiling the Game Logic Directly into the Weights"

### Why It Works
1. **"Can it run Doom?" is internet law** — guaranteed engagement from the meme alone
2. **Clear differentiation from GameNGen** — "deterministic vs. hallucinated" is an easy distinction for anyone to understand
3. **Demonstrates ModelScriptor** — killer demo for the compiler framework
4. **Academic novelty** — first compiled graphical game in transformer weights, publishable result
5. **Karpathy engagement** — he praised Percepta; this is a dramatic escalation of the same idea

### Expected Criticism
- "But why?" → "Because Can It Run Doom is the law. Also: this proves transformers can compute arbitrarily complex deterministic programs, which has implications for understanding what trained transformers are actually doing."
- "Just run the game normally" → "Just use a pregnancy test normally. That's not the point."
- "GameNGen already did this" → "GameNGen hallucinates. This computes. One drifts after 3 seconds. The other runs forever with zero error."

---

## Related Academic Work

| Paper | Year | Relevance |
|-------|------|-----------|
| Thinking Like Transformers (RASP) | ICML 2021 | Foundational language for transformer computation |
| Tracr: Compiled Transformers | 2023 | RASP → transformer weight compiler |
| Looped Transformers as Programmable Computers | ICML 2023 | Constructive proof of transformer universality |
| Diffusion Models Are Real-Time Game Engines (GameNGen) | 2024 | Frame hallucination approach (our foil) |
| Learning Transformer Programs | NeurIPS 2023 | Reverse direction: transformers → programs |
| Learning to Compile Programs to Neural Networks | ICML 2024 | Hypernetwork-based compilation (approximate) |
| ALTA: Compiler-Based Analysis of Transformers | TMLR 2025 | RASP extensions with loops |
| Compiling to Linear Neurons (Cajal) | Nov 2025 | Compiles to differentiable neurons |
| Percepta transformer-vm | Mar 2025 | C → WASM → transformer weights |
| Weights to Code | Jan 2026 | Reverse: extract algorithms from transformer weights |
| Decompiling Transformers to RASP | Feb 2026 | Reverse: trained transformers → RASP programs |

---

## Open Questions

1. **Resolution vs. feasibility:** Can we actually produce a playable frame at 32x32? How many transformer layers does the raycaster require?
2. **Inference speed:** How fast can we run a forward pass? Even 1 FPS would be acceptable for the demo (it's a stunt, not a product).
3. **ModelScriptor readiness:** The compiler needs to be robust enough to handle thousands of operations. Current tests are much smaller scale.
4. **Looping strategy:** Unroll ray-stepping across layers, or find a way to express iteration more compactly?
5. **Demo format:** Interactive (keyboard input, real-time-ish)? Or pre-recorded input sequence rendered as video?
