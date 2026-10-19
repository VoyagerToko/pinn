---
version: 1
slug: "viewer-app-py"
primary_target: "viewer/app.py"
related_targets: ["viewer"]
---

# Surface brief: PINN flow viewer (desktop window)

Scope: the whole viewer window (`viewer/`), one surface with seven scenes. Visitor mode: Experience (the work leads from the first viewport). Audience, job and constraints: see PRODUCT.md (people the author presents to; live predictions next to independent references; failures with equal care).

## Direction contract

THESIS: The window is a marbling bath. Every scene opens by laying pigment on the size, and the trained network's predicted current carries it, live, beside the pulled print of the reference. It refuses the category default: a dark ParaView-style viewer with a scene tree, one viewport and a property panel.

OWN-WORLD: Size-bath grey-green ground (#c9ccbd) around a deeper wet tray; one historic marbling pigment per scene (indigo, malachite, umber, ochre, madder, cobalt violet, verdigris), ink-black text; Bodoni Moda for titles, Libre Franklin with tabular figures for numbers and controls; references framed as pulled sheets with a deckled edge; colour maps interpolated in OKLab from the pigments and kept perceptual; verdicts set as ink stamps that carry the word and the number.

STORY: The audience watches a flow being drawn by the network in real time, sees the truth beside it on the same instant and colour scale, reads the verdict and its number, and for each failure sees the specific evidence (corner divergence, lost flow rate, missing dissipation peak). They leave knowing the networks were trained once and are queried live, accurate where they pass, honest where they fail.

FIRST VIEWPORT: Top rim: seven pigment drops in benchmark order, each with its name and verdict word, the active one struck forward. Body: the bath tray fills about 62% of the width with live pigment, the scene title in Bodoni on its rim and the verdict stamp at its top right; the rack, about 38%, holds the pulled reference print above one evidence chart and a one-sentence caption. Bottom rim: play, a time scrubber showing the whole range (or the Reynolds control with the held-out band hatched), view chips, and a live timing line. Primary action, the signature interaction: click to drop pigment, drag to comb a line of it, and watch the network carry it. Motion grammar: each scene opens with the pigment being laid down in bands before the current takes it; trails fade like dragged ink; everything else moves in 150-250 ms.

FORM: The Marbling Bath, candidate 7 of 7 on the grounded list (assigned by the roll), seed key c4fa4548. Raises: numbers show their source (measured in solid type, live-derived in hairline); every choice stays visible with the active one struck forward; fixed places and precision for every metric; one pigment per scene everywhere; one dominant image per screen with labels on the rim.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

## Platform note

Desktop Qt window (PySide6), not a web page: 2D scenes are painted with QPainter, the 3D scene uses PyVista (pyvistaqt). The web detector does not apply; the finish review judges screenshots against this contract and the craft floor.
