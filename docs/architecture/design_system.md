# CORTEX-M Ecosystem Design System
### Purpose-Driven Chromatic Identities, Visual Environments & Spatial Architecture

> [!NOTE]
> **Design Philosophy**: Technology does not have to default to generic blue and cyan. Each subsystem within the CORTEX ecosystem is assigned a tailored chromatic palette rooted in its purpose, operational personality, and visual environment—ranging from organic earth tones to engineered industrial metals and deep cosmic emissions.

---

## 🎨 The 70 / 20 / 10 Composition Principle

To maintain exceptional visual discipline across CLI tools, documentation, future terminal UI (TUI), and application surfaces, palettes are applied with strict proportions:
* **70% Neutral Foundation**: Background surfaces, deep ink, paper, graphite, or porcelain bases.
* **20% Structural Support**: Secondary panels, boundaries, inactive routes, borders, and structural metadata.
* **10% Signature Accent**: Intentional, high-contrast focal points (active nodes, primary CTA, critical status indicators).

```text
┌────────────────────────────────────────────────────────┐
│ 70% Neutral Foundation (Obsidian / Graphite / Cream)    │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 20% Supporting Structure (Borders / Panels)      │  │
│  │  ┌────────────────────────────────────────────┐  │  │
│  │  │ 10% Signature Accent (Active State Focal)  │  │  │
│  │  └────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

---

## 🌈 The Ten Purpose-Driven Product Palettes

### 1. CORTEX-M (Substrate & Core Runtime) — Obsidian, Copper & Phosphor
* **Inspiration**: Obsidian stone, copper circuitry, phosphorescent organisms, and warm ivory.
* **Personality**: Underlying neural machinery, physical synaptic substrate, organic-mechanical fusion.
* **Role**: Copper highlights physical synaptic connections and Hebbian weights; phosphor serves as a precise signal pulse for active attractor convergence.

| Swatch Name | HEX Code | Suggested Role | Composition Weight |
|---|---|---|---|
| **Obsidian** | `#111310` | Primary deep neutral background | 70% |
| **Graphite Moss** | `#293027` | Structural panels & secondary borders | 20% |
| **Copper** | `#C8784E` | **Primary Signature Accent** (Synaptic brand elements) | 10% |
| **Phosphor** | `#D6E86A` | Live signal pulse / active attractor state | 5% (Micro) |
| **Bone** | `#F3EBDD` | Primary readable foreground text | Contrast Base |

---

### 2. CortexLM (Language Model) — Aubergine, Orchid & Champagne
* **Inspiration**: Dark aubergine, orchid petals, aged archival paper, and champagne gold.
* **Personality**: Expressive, generative, abstract, and literary without conventional AI clichés.
* **Role**: Orchid defines model identity and generative traces; champagne provides warm highlights.

| Swatch Name | HEX Code | Suggested Role | Composition Weight |
|---|---|---|---|
| **Ink Plum** | `#17111A` | Deep contemplative background | 70% |
| **Aubergine** | `#352438` | Contextual cards & structural containers | 20% |
| **Orchid** | `#B77BAF` | **Primary Signature Accent** (Model identity) | 10% |
| **Champagne** | `#E5C46B` | Attractor resonance & selective highlights | 5% (Micro) |
| **Porcelain** | `#F5EBE2` | High-legibility text & token rendering | Contrast Base |

---

### 3. CORTEX-IDE (CortexCode Engineering Workspace) — Forest Graphite & Fern
* **Inspiration**: Dark editor themes, fern leaves, aged brass, and pale mineral surfaces.
* **Personality**: Ergonomic for 12-hour engineering sessions; calm, sharp, and non-distracting.
* **Role**: Fern green provides an organic signature accent; debug brass distinguishes test and review states.

| Swatch Name | HEX Code | Suggested Role | Composition Weight |
|---|---|---|---|
| **Editor Ink** | `#171A17` | Non-glare editor background | 70% |
| **Forest Graphite** | `#29332C` | Gutter, file tree & diff panels | 20% |
| **Fern** | `#8EBF76` | **Primary Signature Accent** (AST symbols & diff additions) | 10% |
| **Debug Brass** | `#E7AF63` | Breakpoints, audit gates & warning highlights | 5% (Micro) |
| **Linen** | `#F0F1E8` | High-contrast monospace code text | Contrast Base |

---

### 4. CORTEX-Personal (Edge Intelligence) — Avocado, Seed & Terracotta
* **Inspiration**: Avocado flesh, its seed, olive leaves, and handmade terracotta pottery.
* **Personality**: Calm, trustworthy, approachable, and intimate; designed for personal daily life.
* **Role**: Light-mode warm cream foundation with deep olive text and terracotta warmth.

| Swatch Name | HEX Code | Suggested Role | Composition Weight |
|---|---|---|---|
| **Cream** | `#F6F0DF` | Warm daylight neutral foundation | 70% |
| **Avocado Flesh** | `#DDE5B6` | Soft contextual grouping surfaces | 20% |
| **Olive Leaf** | `#87985A` | **Primary Signature Accent** (Personal identity & controls) | 10% |
| **Deep Olive** | `#4B603B` | High-contrast typography & structural icons | Contrast Base |
| **Seed Brown** | `#8B5B40` | Terracotta warm accents & memory tags | 5% (Micro) |

---

### 5. CORTEX-Fin (Transaction Replay & Risk Research) — Emerald, Gold & Deep Forest
* **Inspiration**: Emerald gemstones, dark forest foliage, bullion, and archival financial ledgers.
* **Personality**: Institutional credibility, precision, and historical permanence without generic fintech blue.
* **Role**: Emerald anchors brand identity; bullion gold highlights key ratios; profit/loss states remain strictly separate.

| Swatch Name | HEX Code | Suggested Role | Composition Weight |
|---|---|---|---|
| **Forest Ink** | `#101A14` | Shadow-mode execution background | 70% |
| **Evergreen** | `#234B39` | Ledger grouping panels & table headers | 20% |
| **Emerald** | `#3D8B63` | **Primary Signature Accent** (Brand & verified signals) | 10% |
| **Bullion Gold** | `#D1AD58` | Premium metrics, reserve thresholds & highlights | 5% (Micro) |
| **Ledger Cream** | `#F4EBD9` | Numerical ledger text & data tables | Contrast Base |

---

### 6. CORTEX-OS (Cognitive Operating Environment) — Brutalist Graphite & Signal Orange
* **Inspiration**: Industrial architecture, charcoal metal, concrete slabs, and industrial warning paint.
* **Personality**: Structural, utilitarian, bold, and unapologetically engineered for a new computing paradigm.
* **Role**: Neutral concrete surfaces punctuated by high-visibility signal orange for user-space priority.

| Swatch Name | HEX Code | Suggested Role | Composition Weight |
|---|---|---|---|
| **Carbon** | `#131313` | Deep brutalist canvas | 70% |
| **Graphite** | `#292B29` | Window frames, docks & split panes | 20% |
| **Concrete** | `#777B73` | Inactive borders, metadata & secondary icons | 10% |
| **Signal Orange** | `#E56B46` | **Primary Signature Accent** (System attention & omnibar focus) | 5% (Focal) |
| **Acid Lime** | `#C5D75C` | Live process & micro-state indicators | 3% (Micro) |

---

### 7. CORTEX-Robotics (Cyber-Physical Simulation) — Titanium, Safety Orange & Acid Lime
* **Inspiration**: Machined titanium, industrial robot housings, hazard markings, and sensor beacons.
* **Personality**: Heavy-duty, purposeful, safety-certified, and physically grounded.
* **Role**: Machine neutrals build trust; safety orange enforces clear operational boundaries; sensor lime tracks TCP telemetry.

| Swatch Name | HEX Code | Suggested Role | Composition Weight |
|---|---|---|---|
| **Machine Black** | `#181A17` | Workcell simulation viewport background | 70% |
| **Gunmetal** | `#303833` | Kinematic telemetry cards & joint panels | 20% |
| **Titanium** | `#AEB7AB` | Neutral mechanical structural elements & wireframes | 15% |
| **Safety Orange** | `#F07835` | **Primary Signature Accent** (Safety zones & active limits) | 10% |
| **Sensor Lime** | `#D6E85D` | Live TCP path tracking & sensor ping | 5% (Micro) |

---

### 8. CORTEX-Mesh (Decentralized Swarm & Routing) — Moss, Lichen & Signal Lime
* **Inspiration**: Forest-floor moss, lichen colonies, fungal mycelial networks, and branching roots.
* **Personality**: Organic decentralization, distributed resilience, and cooperative civilian logistics.
* **Role**: Dark forest canvas with restrained moss nodes and signal lime pulse for BFT consensus messages.

| Swatch Name | HEX Code | Suggested Role | Composition Weight |
|---|---|---|---|
| **Forest Night** | `#101810` | Swarm network topology canvas | 70% |
| **Moss** | `#2E4B34` | Peer cluster containers & inactive routing edges | 20% |
| **Lichen Green** | `#729775` | **Primary Signature Accent** (Active node status) | 10% |
| **Signal Lime** | `#CBE86B` | High-priority routing packet pulse | 5% (Micro) |
| **Parchment** | `#F1EBD7` | Swarm telemetry typography | Contrast Base |

---

### 9. CORTEX-Health / BCI (Neural Signal Research) — Jade, Porcelain & Botanical Sage
* **Inspiration**: Celadon ceramics, raw jade, medicinal herbariums, and clinical porcelain.
* **Personality**: Human, restorative, tranquil, and scientifically rigorous without cold hospital sterility.
* **Role**: Clean porcelain background with soothing botanical sage waves and rich jade accents.

| Swatch Name | HEX Code | Suggested Role | Composition Weight |
|---|---|---|---|
| **Porcelain** | `#F6F1E5` | Clean, non-glare scientific background | 70% |
| **Celadon** | `#D4E2D0` | Waveform canvas backdrops & channel dividers | 20% |
| **Sage** | `#A6C6A6` | Microvolt EEG traces & baseline waves | 15% |
| **Jade** | `#689785` | **Primary Signature Accent** (Bandpower peaks & cognitive targets) | 10% |
| **Deep Botanical**| `#315C46` | High-legibility clinical typography | Contrast Base |

---

### 10. CORTEX-Space (Orbital Telemetry & Simulation) — Cosmic Ember & Burnished Gold
* **Inspiration**: Dark emission nebulae, stellar hydrogen clouds, cosmic dust, and distant starlight.
* **Personality**: Deep astronomical awe, mission-critical precision, and quiet orbital focus.
* **Role**: Void black with cosmic burgundy orbital tracks, solar ember telemetry alerts, and starlight gold ephemeris.

| Swatch Name | HEX Code | Suggested Role | Composition Weight |
|---|---|---|---|
| **Void** | `#121014` | Deep space ephemeris background | 70% |
| **Nebula Plum** | `#33202E` | Orbital ribbon planes & telemetry trays | 20% |
| **Cosmic Burgundy** | `#86455D` | **Primary Signature Accent** (Mission trajectory traces) | 10% |
| **Solar Ember** | `#E77C53` | Solar array alerts & thermal anomaly highlights | 5% (Micro) |
| **Starlight Gold**| `#E9BD73` | Ground station line-of-sight & coordinate axes | 5% (Micro) |

---

## 🧭 Ecosystem Signature Summary

| Product System | Primary Accent | Accent HEX | Atmospheric Domain |
|---|---|---|---|
| **CORTEX-M** | **Copper** | `#C8784E` | Neural Machinery & In-Memory Plasticity |
| **CortexLM** | **Orchid** | `#B77BAF` | Generative Cognition & Latent Intent |
| **CORTEX-IDE** | **Fern** | `#8EBF76` | Developer Ergonomics & AST Refactoring |
| **CORTEX-Personal** | **Olive** | `#87985A` | Approachable Private Edge Cognition |
| **CORTEX-Fin** | **Emerald** | `#3D8B63` | Archival Ledgers & Shadow-Mode Analytics |
| **CORTEX-OS** | **Signal Orange** | `#E56B46` | Brutalist Userspace System Orchestration |
| **CORTEX-Robotics** | **Safety Orange** | `#F07835` | Cyber-Physical Control & Safety Interlocks |
| **CORTEX-Mesh** | **Lichen** | `#729775` | Decentralized Swarm Mycelium & Consensus |
| **CORTEX-Health** | **Jade** | `#689785` | Restorative Non-Clinical Neural Signal Decoding |
| **CORTEX-Space** | **Cosmic Burgundy** | `#86455D` | Deep-Space Ephemeris & Telemetry Streams |

---

## 📐 Semantic Isolation Rule
Brand identity colors are **never** overloaded to represent deterministic system status:
* **Evidence Statuses** (`VERIFIED`, `PARTIALLY_SUPPORTED`, `CONFLICTING_EVIDENCE`, `INSUFFICIENT_EVIDENCE`) use consistent mathematical semantic indicators with independent icons and text contrast checks.
* **Safety / Fault States** (E-Stop, Margin Violation, Packet Drop) always include explicit symbols and text labels, ensuring full accessibility regardless of color perception.
