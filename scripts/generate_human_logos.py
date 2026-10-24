"""Generate human-centric, minimalist vector and high-resolution raster logos for Cortex-M.

Design Philosophy:
- Replaces stereotypical AI 'gradient soup' with human semiotic graphic design (Paul Rand, Dieter Rams, Massimo Vignelli).
- Rooted in Gestalt psychology: Closure, Continuity, Figure-Ground, and Semantic Functionality.
- Scalable from 16x16 favicon to 4K display without loss of legibility.
- Pure vector SVGs and crisp 300-DPI PNGs.
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "branding"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. SVG GENERATION (Pure Vector Code)
# =============================================================================

def build_cortex_m_svg() -> str:
    """CORTEX-M: The Cortical M - Neocortex sulci forming the letter M with MIMO pathways."""
    return '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="100%" height="100%">
  <defs>
    <linearGradient id="mBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0B0F19"/>
      <stop offset="100%" stop-color="#070A10"/>
    </linearGradient>
    <linearGradient id="mPath" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#2563EB"/>
      <stop offset="50%" stop-color="#3B82F6"/>
      <stop offset="100%" stop-color="#06B6D4"/>
    </linearGradient>
    <filter id="subtleGlow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="8" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
  </defs>

  <!-- Canvas Container -->
  <rect width="512" height="512" rx="96" fill="url(#mBg)"/>
  <rect width="508" height="508" x="2" y="2" rx="94" fill="none" stroke="#1E293B" stroke-width="2"/>

  <!-- Subtle Coordinate Grid Substrate (Zero-DB Architecture) -->
  <g opacity="0.07" stroke="#94A3B8" stroke-width="1">
    <line x1="128" y1="64" x2="128" y2="448"/>
    <line x1="256" y1="64" x2="256" y2="448"/>
    <line x1="384" y1="64" x2="384" y2="448"/>
    <line x1="64" y1="128" x2="448" y2="128"/>
    <line x1="64" y1="256" x2="448" y2="256"/>
    <line x1="64" y1="384" x2="448" y2="384"/>
  </g>

  <!-- MIMO Secondary Resonance Pathway (Dorsal Channel) -->
  <path d="M 120 340 
           C 120 230, 140 180, 180 180 
           C 220 180, 240 240, 256 270 
           C 272 240, 292 180, 332 180 
           C 372 180, 392 230, 392 340"
        fill="none" 
        stroke="#1E3A8A" 
        stroke-width="8" 
        stroke-linecap="round" 
        stroke-dasharray="4 12" 
        opacity="0.6"/>

  <!-- Primary Cortical M Pathway (Ventral Stream / Hopfield Manifold) -->
  <path d="M 104 350 
           C 104 200, 130 140, 184 140 
           C 228 140, 246 220, 256 250 
           C 266 220, 284 140, 328 140 
           C 382 140, 408 200, 408 350"
        fill="none" 
        stroke="url(#mPath)" 
        stroke-width="22" 
        stroke-linecap="round" 
        stroke-linejoin="round"
        filter="url(#subtleGlow)"/>

  <!-- Synaptic Attractor Nodes (Energy Wells / Fixed Points) -->
  <!-- Left Ingress Node -->
  <circle cx="104" cy="350" r="14" fill="#3B82F6"/>
  <circle cx="104" cy="350" r="6" fill="#FFFFFF"/>

  <!-- Left Cortical Crest -->
  <circle cx="184" cy="140" r="10" fill="#06B6D4"/>
  <circle cx="184" cy="140" r="4" fill="#FFFFFF"/>

  <!-- Central Hopfield Attractor (Convergence Minimum) -->
  <circle cx="256" cy="250" r="16" fill="#F59E0B"/>
  <circle cx="256" cy="250" r="7" fill="#FFFFFF"/>

  <!-- Right Cortical Crest -->
  <circle cx="328" cy="140" r="10" fill="#06B6D4"/>
  <circle cx="328" cy="140" r="4" fill="#FFFFFF"/>

  <!-- Right Egress Node -->
  <circle cx="408" cy="350" r="14" fill="#3B82F6"/>
  <circle cx="408" cy="350" r="6" fill="#FFFFFF"/>

  <!-- Brand Typography -->
  <text x="256" y="420" 
        text-anchor="middle" 
        font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', sans-serif" 
        font-size="34" 
        font-weight="800" 
        letter-spacing="6" 
        fill="#F8FAFC">CORTEX-M</text>

  <text x="256" y="448" 
        text-anchor="middle" 
        font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', sans-serif" 
        font-size="12" 
        font-weight="600" 
        letter-spacing="4" 
        fill="#64748B">CONTINUOUS SYNAPTIC MIMO SUBSTRATE</text>
</svg>'''


def build_cortex_code_svg() -> str:
    """CORTEX-Coder: The Neural Brackets - Code braces forming neocortex hemispheres with AST nodes."""
    return '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="100%" height="100%">
  <defs>
    <linearGradient id="codeBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0D1117"/>
      <stop offset="100%" stop-color="#080C10"/>
    </linearGradient>
    <linearGradient id="codePath" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#059669"/>
      <stop offset="50%" stop-color="#10B981"/>
      <stop offset="100%" stop-color="#34D399"/>
    </linearGradient>
    <filter id="codeGlow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="7" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
  </defs>

  <!-- Container -->
  <rect width="512" height="512" rx="96" fill="url(#codeBg)"/>
  <rect width="508" height="508" x="2" y="2" rx="94" fill="none" stroke="#1E293B" stroke-width="2"/>

  <!-- Background Terminal Scanlines -->
  <g opacity="0.05" stroke="#10B981" stroke-width="1">
    <line x1="64" y1="160" x2="448" y2="160"/>
    <line x1="64" y1="200" x2="448" y2="200"/>
    <line x1="64" y1="240" x2="448" y2="240"/>
    <line x1="64" y1="280" x2="448" y2="280"/>
    <line x1="64" y1="320" x2="448" y2="320"/>
  </g>

  <!-- Left Hemisphere Code Brace '{' (Architect + Coder) -->
  <path d="M 210 130 
           C 160 130, 140 160, 140 200 
           C 140 235, 128 245, 110 250 
           C 128 255, 140 265, 140 300 
           C 140 340, 160 370, 210 370"
        fill="none" 
        stroke="url(#codePath)" 
        stroke-width="18" 
        stroke-linecap="round"
        filter="url(#codeGlow)"/>

  <!-- Right Hemisphere Code Brace '}' (Critic + Auditor) -->
  <path d="M 302 130 
           C 352 130, 372 160, 372 200 
           C 372 235, 384 245, 402 250 
           C 384 255, 372 265, 372 300 
           C 372 340, 352 370, 302 370"
        fill="none" 
        stroke="url(#codePath)" 
        stroke-width="18" 
        stroke-linecap="round"
        filter="url(#codeGlow)"/>

  <!-- Central Synaptic Code Core: Execution Prompt '>' and Block '_' -->
  <path d="M 236 215 L 272 250 L 236 285" 
        fill="none" 
        stroke="#34D399" 
        stroke-width="12" 
        stroke-linecap="round" 
        stroke-linejoin="round"/>

  <!-- Terminal Block Cursor (Verified Execution) -->
  <rect x="276" y="278" width="16" height="8" rx="2" fill="#10B981"/>

  <!-- 4 Multi-Agent Swarm Nodes -->
  <!-- Top Left: Architect (DAG Design) -->
  <circle cx="210" cy="130" r="10" fill="#8B5CF6"/>
  <circle cx="210" cy="130" r="4" fill="#FFFFFF"/>

  <!-- Bottom Left: Coder (AST Synthesis) -->
  <circle cx="210" cy="370" r="10" fill="#3B82F6"/>
  <circle cx="210" cy="370" r="4" fill="#FFFFFF"/>

  <!-- Top Right: Critic (Static Analysis & Risk Guard) -->
  <circle cx="302" cy="130" r="10" fill="#F59E0B"/>
  <circle cx="302" cy="130" r="4" fill="#FFFFFF"/>

  <!-- Bottom Right: Auditor (Runtime Sandbox Verified) -->
  <circle cx="302" cy="370" r="10" fill="#10B981"/>
  <circle cx="302" cy="370" r="4" fill="#FFFFFF"/>

  <!-- Brand Typography -->
  <text x="256" y="420" 
        text-anchor="middle" 
        font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', monospace" 
        font-size="32" 
        font-weight="800" 
        letter-spacing="5" 
        fill="#F8FAFC">CORTEX-CODE</text>

  <text x="256" y="448" 
        text-anchor="middle" 
        font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', monospace" 
        font-size="12" 
        font-weight="600" 
        letter-spacing="4" 
        fill="#64748B">AUTONOMOUS MULTI-AGENT SWARM</text>
</svg>'''


def build_cortex_lm_svg() -> str:
    """CORTEX-LM: The Dialogue Synapse - Speech bubble merging into cortical harmonic waves."""
    return '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="100%" height="100%">
  <defs>
    <linearGradient id="lmBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0F172A"/>
      <stop offset="100%" stop-color="#090D16"/>
    </linearGradient>
    <linearGradient id="lmPath" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#7C3AED"/>
      <stop offset="50%" stop-color="#8B5CF6"/>
      <stop offset="100%" stop-color="#C084FC"/>
    </linearGradient>
    <filter id="lmGlow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="8" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
  </defs>

  <!-- Container -->
  <rect width="512" height="512" rx="96" fill="url(#lmBg)"/>
  <rect width="508" height="508" x="2" y="2" rx="94" fill="none" stroke="#1E293B" stroke-width="2"/>

  <!-- Minimalist Dialogue Balloon with Cortical Tail -->
  <path d="M 120 150 
           A 60 60 0 0 1 392 150 
           L 392 270 
           A 60 60 0 0 1 332 330 
           L 190 330 
           L 130 370 
           L 130 320 
           A 60 60 0 0 1 120 270 Z" 
        fill="#131B2E" 
        stroke="#334155" 
        stroke-width="3"/>

  <!-- Internal Harmonic Cortical Wave 1 (Upper Association Wave) -->
  <path d="M 160 200 
           C 190 180, 220 220, 256 200 
           C 292 180, 322 220, 352 200"
        fill="none" 
        stroke="url(#lmPath)" 
        stroke-width="12" 
        stroke-linecap="round"
        filter="url(#lmGlow)"/>

  <!-- Internal Harmonic Cortical Wave 2 (Zero-Hallucination Energy Ground) -->
  <path d="M 175 250 
           C 205 235, 230 265, 256 250 
           C 282 235, 310 265, 337 250"
        fill="none" 
        stroke="#6366F1" 
        stroke-width="10" 
        stroke-linecap="round"/>

  <!-- Synaptic Resonance Nodes along language stream -->
  <circle cx="256" cy="200" r="10" fill="#C084FC"/>
  <circle cx="256" cy="200" r="4" fill="#FFFFFF"/>

  <circle cx="352" cy="200" r="8" fill="#8B5CF6"/>
  <circle cx="352" cy="200" r="3" fill="#FFFFFF"/>

  <circle cx="160" cy="200" r="8" fill="#7C3AED"/>
  <circle cx="160" cy="200" r="3" fill="#FFFFFF"/>

  <!-- Dialogue Focus Anchor Point -->
  <circle cx="130" cy="370" r="6" fill="#A855F7"/>

  <!-- Brand Typography -->
  <text x="256" y="420" 
        text-anchor="middle" 
        font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', sans-serif" 
        font-size="34" 
        font-weight="800" 
        letter-spacing="6" 
        fill="#F8FAFC">CORTEX-LM</text>

  <text x="256" y="448" 
        text-anchor="middle" 
        font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', sans-serif" 
        font-size="12" 
        font-weight="600" 
        letter-spacing="4" 
        fill="#64748B">BIO-INSPIRED NEURAL LANGUAGE MODEL</text>
</svg>'''


def build_ecosystem_banner_svg() -> str:
    """Unified ecosystem brand showcase with the 3 identities side-by-side."""
    return '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 500" width="100%" height="100%">
  <defs>
    <linearGradient id="suiteBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0B0F19"/>
      <stop offset="50%" stop-color="#080C14"/>
      <stop offset="100%" stop-color="#05080E"/>
    </linearGradient>
  </defs>

  <rect width="1440" height="500" rx="32" fill="url(#suiteBg)"/>
  <rect width="1436" height="496" x="2" y="2" rx="30" fill="none" stroke="#1E293B" stroke-width="2"/>

  <!-- Header Header -->
  <text x="720" y="65" text-anchor="middle" font-family="'Inter', -apple-system, sans-serif" font-size="26" font-weight="800" letter-spacing="8" fill="#F8FAFC">CORTEX-M ECOSYSTEM SUITE</text>
  <text x="720" y="92" text-anchor="middle" font-family="'Inter', -apple-system, sans-serif" font-size="13" font-weight="500" letter-spacing="4" fill="#64748B">BIO-INSPIRED NEURAL FOUNDATIONS • AUTONOMOUS AGENTIC SWARMS • ZERO-DATABASE MEMORY</text>

  <!-- 3 Brand Columns -->
  <!-- Column 1: CORTEX-M Core Substrate -->
  <g transform="translate(100, 120)">
    <rect width="360" height="320" rx="24" fill="#0E1424" stroke="#1E293B" stroke-width="1.5"/>
    <!-- Icon mini -->
    <path d="M 120 180 C 120 100, 140 60, 170 60 C 195 60, 205 110, 210 130 C 215 110, 225 60, 250 60 C 280 60, 300 100, 300 180" fill="none" stroke="#3B82F6" stroke-width="12" stroke-linecap="round"/>
    <circle cx="120" cy="180" r="8" fill="#3B82F6"/>
    <circle cx="170" cy="60" r="7" fill="#06B6D4"/>
    <circle cx="210" cy="130" r="9" fill="#F59E0B"/>
    <circle cx="250" cy="60" r="7" fill="#06B6D4"/>
    <circle cx="300" cy="180" r="8" fill="#3B82F6"/>

    <text x="180" y="235" text-anchor="middle" font-family="'Inter', sans-serif" font-size="20" font-weight="800" letter-spacing="4" fill="#F8FAFC">CORTEX-M</text>
    <text x="180" y="260" text-anchor="middle" font-family="'Inter', sans-serif" font-size="11" font-weight="600" letter-spacing="2" fill="#3B82F6">NEURAL SUBSTRATE &amp; MIMO</text>
    <text x="180" y="285" text-anchor="middle" font-family="'Inter', sans-serif" font-size="10" font-weight="400" fill="#94A3B8">Continuous Hopfield (W) • Zero-DB</text>
  </g>

  <!-- Column 2: CORTEX-Coder (CortexCode) -->
  <g transform="translate(540, 120)">
    <rect width="360" height="320" rx="24" fill="#0E1815" stroke="#134E4A" stroke-width="1.5"/>
    <!-- Icon mini -->
    <path d="M 155 60 C 125 60, 115 80, 115 110 C 115 130, 105 140, 95 145 C 105 150, 115 160, 115 180 C 115 210, 125 230, 155 230" fill="none" stroke="#10B981" stroke-width="10" stroke-linecap="round"/>
    <path d="M 205 60 C 235 60, 245 80, 245 110 C 245 130, 255 140, 265 145 C 255 150, 245 160, 245 180 C 245 210, 235 230, 205 230" fill="none" stroke="#10B981" stroke-width="10" stroke-linecap="round"/>
    <path d="M 170 125 L 190 145 L 170 165" fill="none" stroke="#34D399" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="155" cy="60" r="6" fill="#8B5CF6"/>
    <circle cx="155" cy="230" r="6" fill="#3B82F6"/>
    <circle cx="205" cy="60" r="6" fill="#F59E0B"/>
    <circle cx="205" cy="230" r="6" fill="#10B981"/>

    <text x="180" y="235" text-anchor="middle" font-family="'Inter', sans-serif" font-size="20" font-weight="800" letter-spacing="4" fill="#F8FAFC">CORTEX-CODER</text>
    <text x="180" y="260" text-anchor="middle" font-family="'Inter', sans-serif" font-size="11" font-weight="600" letter-spacing="2" fill="#10B981">AUTONOMOUS CODING SWARM</text>
    <text x="180" y="285" text-anchor="middle" font-family="'Inter', sans-serif" font-size="10" font-weight="400" fill="#94A3B8">Architect • Coder • Critic • Auditor</text>
  </g>

  <!-- Column 3: CORTEX-LM -->
  <g transform="translate(980, 120)">
    <rect width="360" height="320" rx="24" fill="#151226" stroke="#4C1D95" stroke-width="1.5"/>
    <!-- Icon mini -->
    <path d="M 110 80 A 30 30 0 0 1 250 80 L 250 160 A 30 30 0 0 1 220 190 L 140 190 L 115 210 L 115 190 A 30 30 0 0 1 110 160 Z" fill="#1F1A3A" stroke="#475569" stroke-width="2"/>
    <path d="M 135 125 C 150 115, 170 135, 185 125 C 200 115, 215 135, 225 125" fill="none" stroke="#8B5CF6" stroke-width="8" stroke-linecap="round"/>
    <circle cx="185" cy="125" r="7" fill="#C084FC"/>
    <circle cx="135" cy="125" r="5" fill="#7C3AED"/>
    <circle cx="225" cy="125" r="5" fill="#8B5CF6"/>

    <text x="180" y="235" text-anchor="middle" font-family="'Inter', sans-serif" font-size="20" font-weight="800" letter-spacing="4" fill="#F8FAFC">CORTEX-LM</text>
    <text x="180" y="260" text-anchor="middle" font-family="'Inter', sans-serif" font-size="11" font-weight="600" letter-spacing="2" fill="#8B5CF6">NEURAL LANGUAGE MODEL</text>
    <text x="180" y="285" text-anchor="middle" font-family="'Inter', sans-serif" font-size="10" font-weight="400" fill="#94A3B8">Lyapunov Guard • Associative Memory</text>
  </g>
</svg>'''


# =============================================================================
# 2. RASTER PNG RENDERING (Anti-Aliased, High Resolution)
# =============================================================================

def render_cortex_m_png(output_path: str, size: int = 1024):
    """Render pixel-perfect, mathematically computed CORTEX-M brand logo."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=size // 8)
    fig.patch.set_facecolor("#0B0F19")
    ax.set_facecolor("#0B0F19")
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.axis("off")

    # Container outline
    rect = patches.FancyBboxPatch((8, 8), 496, 496, boxstyle="round,pad=0,rounding_size=96",
                                  linewidth=2, edgecolor="#1E293B", facecolor="#0B0F19")
    ax.add_patch(rect)

    # Substrate Grid
    for x in [128, 256, 384]:
        ax.plot([x, x], [64, 448], color="#94A3B8", alpha=0.06, lw=1)
    for y in [128, 256, 384]:
        ax.plot([64, 448], [y, y], color="#94A3B8", alpha=0.06, lw=1)

    # Mathematical Cortical M Curve (Bézier spline)
    # In matplotlib coordinates, y=0 is bottom, so invert y from SVG
    # SVG y: 350 -> MPL y: 162; SVG y: 140 -> MPL y: 372; SVG y: 250 -> MPL y: 262
    verts = [
        (104, 162),  # Ingress
        (104, 312), (130, 372), (184, 372), # Crest 1
        (228, 372), (246, 292), (256, 262), # Valley (Hopfield Attractor)
        (266, 292), (284, 372), (328, 372), # Crest 2
        (382, 372), (408, 312), (408, 162), # Egress
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
    ]
    path = Path(verts, codes)

    # Secondary pathway (Dorsal)
    verts_dorsal = [
        (120, 172),
        (120, 282), (140, 332), (180, 332),
        (220, 332), (240, 272), (256, 242),
        (272, 272), (292, 332), (332, 332),
        (372, 332), (392, 282), (392, 172),
    ]
    path_dorsal = Path(verts_dorsal, codes)
    patch_dorsal = patches.PathPatch(path_dorsal, facecolor='none', edgecolor='#1E3A8A',
                                     lw=5, linestyle='--', alpha=0.6, capstyle='round')
    ax.add_patch(patch_dorsal)

    # Primary pathway
    patch_primary = patches.PathPatch(path, facecolor='none', edgecolor='#3B82F6',
                                      lw=16, capstyle='round', joinstyle='round')
    ax.add_patch(patch_primary)

    # Attractor Nodes
    nodes = [
        (104, 162, '#3B82F6', 10),
        (184, 372, '#06B6D4', 8),
        (256, 262, '#F59E0B', 12),  # Central attractor
        (328, 372, '#06B6D4', 8),
        (408, 162, '#3B82F6', 10),
    ]
    for x, y, col, r in nodes:
        ax.add_patch(plt.Circle((x, y), r, color=col, zorder=5))
        ax.add_patch(plt.Circle((x, y), r * 0.4, color='#FFFFFF', zorder=6))

    # Clean Typography
    ax.text(256, 92, "CORTEX-M", ha="center", va="center",
            fontsize=26, fontweight="bold", color="#F8FAFC", family="sans-serif")
    ax.text(256, 64, "CONTINUOUS SYNAPTIC MIMO SUBSTRATE", ha="center", va="center",
            fontsize=8.5, fontweight="semibold", color="#64748B", family="sans-serif")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=size // 8, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()


def render_cortex_code_png(output_path: str, size: int = 1024):
    """Render pixel-perfect, mathematically computed CORTEX-Coder brand logo."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=size // 8)
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.axis("off")

    # Container outline
    rect = patches.FancyBboxPatch((8, 8), 496, 496, boxstyle="round,pad=0,rounding_size=96",
                                  linewidth=2, edgecolor="#1E293B", facecolor="#0D1117")
    ax.add_patch(rect)

    # Left Brace '{'
    verts_left = [
        (210, 382),
        (160, 382), (140, 352), (140, 312),
        (140, 277), (128, 267), (110, 262),
        (128, 257), (140, 247), (140, 212),
        (140, 172), (160, 142), (210, 142)
    ]
    codes_brace = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4
    ]
    p_left = Path(verts_left, codes_brace)
    ax.add_patch(patches.PathPatch(p_left, facecolor='none', edgecolor='#10B981', lw=15, capstyle='round'))

    # Right Brace '}'
    verts_right = [
        (302, 382),
        (352, 382), (372, 352), (372, 312),
        (372, 277), (384, 267), (402, 262),
        (384, 257), (372, 247), (372, 212),
        (372, 172), (352, 142), (302, 142)
    ]
    p_right = Path(verts_right, codes_brace)
    ax.add_patch(patches.PathPatch(p_right, facecolor='none', edgecolor='#10B981', lw=15, capstyle='round'))

    # Central execution prompt '>'
    ax.plot([236, 272, 236], [297, 262, 227], color='#34D399', lw=10, solid_capstyle='round', solid_joinstyle='round')
    # Terminal block cursor
    ax.add_patch(patches.Rectangle((276, 226), 16, 8, facecolor='#10B981'))

    # 4 Swarm Nodes
    swarm_nodes = [
        (210, 382, '#8B5CF6'),  # Architect
        (210, 142, '#3B82F6'),  # Coder
        (302, 382, '#F59E0B'),  # Critic
        (302, 142, '#10B981'),  # Auditor
    ]
    for x, y, col in swarm_nodes:
        ax.add_patch(plt.Circle((x, y), 8, color=col, zorder=5))
        ax.add_patch(plt.Circle((x, y), 3, color='#FFFFFF', zorder=6))

    # Typography
    ax.text(256, 92, "CORTEX-CODE", ha="center", va="center",
            fontsize=26, fontweight="bold", color="#F8FAFC", family="monospace")
    ax.text(256, 64, "AUTONOMOUS MULTI-AGENT SWARM", ha="center", va="center",
            fontsize=8.5, fontweight="semibold", color="#64748B", family="monospace")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=size // 8, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()


def render_cortex_lm_png(output_path: str, size: int = 1024):
    """Render pixel-perfect, mathematically computed CORTEX-LM brand logo."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=size // 8)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.axis("off")

    # Container outline
    rect = patches.FancyBboxPatch((8, 8), 496, 496, boxstyle="round,pad=0,rounding_size=96",
                                  linewidth=2, edgecolor="#1E293B", facecolor="#0F172A")
    ax.add_patch(rect)

    # Speech Bubble
    bubble_box = patches.FancyBboxPatch((120, 182), 272, 180, boxstyle="round,pad=0,rounding_size=50",
                                        linewidth=2.5, edgecolor="#334155", facecolor="#131B2E")
    ax.add_patch(bubble_box)
    # Speech tail triangle
    tail = patches.Polygon([[130, 192], [190, 182], [130, 142]], closed=True, facecolor="#131B2E", edgecolor="#334155", lw=2)
    ax.add_patch(tail)

    # Harmonic Cortical Wave 1 (Upper Association Wave)
    verts_wave1 = [
        (160, 312),
        (190, 332), (220, 292), (256, 312),
        (292, 332), (322, 292), (352, 312)
    ]
    codes_wave = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4
    ]
    p_wave1 = Path(verts_wave1, codes_wave)
    ax.add_patch(patches.PathPatch(p_wave1, facecolor='none', edgecolor='#8B5CF6', lw=10, capstyle='round'))

    # Harmonic Cortical Wave 2 (Zero-Hallucination Ground)
    verts_wave2 = [
        (175, 262),
        (205, 277), (230, 247), (256, 262),
        (282, 277), (310, 247), (337, 262)
    ]
    p_wave2 = Path(verts_wave2, codes_wave)
    ax.add_patch(patches.PathPatch(p_wave2, facecolor='none', edgecolor='#6366F1', lw=8, capstyle='round'))

    # Synaptic resonance nodes
    ax.add_patch(plt.Circle((256, 312), 8, color='#C084FC', zorder=5))
    ax.add_patch(plt.Circle((256, 312), 3, color='#FFFFFF', zorder=6))

    ax.add_patch(plt.Circle((160, 312), 6, color='#7C3AED', zorder=5))
    ax.add_patch(plt.Circle((352, 312), 6, color='#8B5CF6', zorder=5))

    # Typography
    ax.text(256, 92, "CORTEX-LM", ha="center", va="center",
            fontsize=26, fontweight="bold", color="#F8FAFC", family="sans-serif")
    ax.text(256, 64, "BIO-INSPIRED NEURAL LANGUAGE MODEL", ha="center", va="center",
            fontsize=8.5, fontweight="semibold", color="#64748B", family="sans-serif")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=size // 8, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()


def render_ecosystem_banner_png(output_path: str):
    """Render unified horizontal 1440x500 banner with the 3 identities."""
    fig, ax = plt.subplots(figsize=(14.4, 5.0), dpi=100)
    fig.patch.set_facecolor("#0B0F19")
    ax.set_facecolor("#0B0F19")
    ax.set_xlim(0, 1440)
    ax.set_ylim(0, 500)
    ax.axis("off")

    # Banner border
    rect = patches.FancyBboxPatch((8, 8), 1424, 484, boxstyle="round,pad=0,rounding_size=32",
                                  linewidth=2, edgecolor="#1E293B", facecolor="#0B0F19")
    ax.add_patch(rect)

    # Header
    ax.text(720, 435, "CORTEX-M ECOSYSTEM SUITE", ha="center", va="center",
            fontsize=20, fontweight="bold", color="#F8FAFC", family="sans-serif")
    ax.text(720, 408, "BIO-INSPIRED NEURAL FOUNDATIONS • AUTONOMOUS AGENTIC SWARMS • ZERO-DATABASE MEMORY",
            ha="center", va="center", fontsize=9.5, fontweight="semibold", color="#64748B", family="sans-serif")

    # 3 Columns
    col_configs = [
        (100, 60, "CORTEX-M", "NEURAL SUBSTRATE & MIMO", "Continuous Hopfield (W) • Zero-DB", "#0E1424", "#1E293B", "#3B82F6"),
        (540, 60, "CORTEX-CODER", "AUTONOMOUS CODING SWARM", "Architect • Coder • Critic • Auditor", "#0E1815", "#134E4A", "#10B981"),
        (980, 60, "CORTEX-LM", "NEURAL LANGUAGE MODEL", "Lyapunov Guard • Associative Memory", "#151226", "#4C1D95", "#8B5CF6")
    ]

    for x_offset, y_offset, title, subtitle, desc, card_bg, card_border, accent in col_configs:
        card = patches.FancyBboxPatch((x_offset, y_offset), 360, 310, boxstyle="round,pad=0,rounding_size=24",
                                      linewidth=1.5, edgecolor=card_border, facecolor=card_bg)
        ax.add_patch(card)

        ax.text(x_offset + 180, y_offset + 95, title, ha="center", va="center",
                fontsize=16, fontweight="bold", color="#F8FAFC", family="sans-serif")
        ax.text(x_offset + 180, y_offset + 70, subtitle, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=accent, family="sans-serif")
        ax.text(x_offset + 180, y_offset + 45, desc, ha="center", va="center",
                fontsize=8.5, fontweight="normal", color="#94A3B8", family="sans-serif")

    # Column 1 Mini Symbol: The Cortical M
    verts_m = [
        (220, 200),
        (220, 280), (240, 320), (270, 320),
        (295, 320), (305, 270), (310, 250),
        (315, 270), (325, 320), (350, 320),
        (380, 320), (400, 280), (400, 200)
    ]
    codes_m = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    ax.add_patch(patches.PathPatch(Path(verts_m, codes_m), facecolor='none', edgecolor='#3B82F6', lw=8, capstyle='round'))
    ax.add_patch(plt.Circle((310, 250), 6, color='#F59E0B'))
    ax.add_patch(plt.Circle((270, 320), 5, color='#06B6D4'))
    ax.add_patch(plt.Circle((350, 320), 5, color='#06B6D4'))

    # Column 2 Mini Symbol: The Brackets
    verts_b1 = [(695, 320), (665, 320), (655, 300), (655, 270), (655, 250), (645, 240), (635, 235), (645, 230), (655, 220), (655, 200), (655, 170), (665, 150), (695, 150)]
    verts_b2 = [(745, 320), (775, 320), (785, 300), (785, 270), (785, 250), (795, 240), (805, 235), (795, 230), (785, 220), (785, 200), (785, 170), (775, 150), (745, 150)]
    ax.add_patch(patches.PathPatch(Path(verts_b1, codes_m), facecolor='none', edgecolor='#10B981', lw=7, capstyle='round'))
    ax.add_patch(patches.PathPatch(Path(verts_b2, codes_m), facecolor='none', edgecolor='#10B981', lw=7, capstyle='round'))
    ax.plot([710, 730, 710], [255, 235, 215], color='#34D399', lw=6, solid_capstyle='round')

    # Column 3 Mini Symbol: Speech Synapse
    ax.add_patch(patches.FancyBboxPatch((1100, 185), 120, 85, boxstyle="round,pad=0,rounding_size=25", facecolor="#1F1A3A", edgecolor="#475569", lw=1.5))
    verts_w = [(1120, 225), (1135, 235), (1150, 215), (1165, 225), (1180, 235), (1195, 215), (1205, 225)]
    codes_w = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    ax.add_patch(patches.PathPatch(Path(verts_w, codes_w), facecolor='none', edgecolor='#8B5CF6', lw=6, capstyle='round'))
    ax.add_patch(plt.Circle((1165, 225), 5, color='#C084FC'))

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=100, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()


def main():
    print("[1/4] Writing precision vector SVGs...")
    with open(os.path.join(OUTPUT_DIR, "cortex_m_logo.svg"), "w", encoding="utf-8") as f:
        f.write(build_cortex_m_svg())
    with open(os.path.join(OUTPUT_DIR, "cortex_code_logo.svg"), "w", encoding="utf-8") as f:
        f.write(build_cortex_code_svg())
    with open(os.path.join(OUTPUT_DIR, "cortex_lm_logo.svg"), "w", encoding="utf-8") as f:
        f.write(build_cortex_lm_svg())
    with open(os.path.join(OUTPUT_DIR, "cortex_brand_ecosystem.svg"), "w", encoding="utf-8") as f:
        f.write(build_ecosystem_banner_svg())

    print("[2/4] Rendering crisp 1024x1024 PNGs...")
    render_cortex_m_png(os.path.join(OUTPUT_DIR, "cortex_m_logo.png"), size=1024)
    render_cortex_code_png(os.path.join(OUTPUT_DIR, "cortex_code_logo.png"), size=1024)
    render_cortex_lm_png(os.path.join(OUTPUT_DIR, "cortex_lm_logo.png"), size=1024)

    print("[3/4] Rendering unified ecosystem banner PNG (1440x500)...")
    render_ecosystem_banner_png(os.path.join(OUTPUT_DIR, "cortex_brand_ecosystem.png"))

    print("[4/4] Removing legacy AI gradient-soup JPEG files...")
    for old_file in ["cortex_m_logo.jpg", "cortex_code_logo.jpg", "cortex_lm_logo.jpg"]:
        old_path = os.path.join(OUTPUT_DIR, old_file)
        if os.path.exists(old_path):
            os.remove(old_path)
            print(f"  Removed {old_file}")

    print(f"\nAll human-designed brand identity assets generated successfully in:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
