---
name: pinnflow viewer
description: The trained flows, live. A marbling bath in which a physics-informed network's predicted current carries the pigment.
colors:
  size: "#c9ccbd"
  size-wet: "#b7bdab"
  size-deep: "#a2a993"
  rim: "#d5d7ca"
  paper: "#e6e3d6"
  stone: "#8d927f"
  ink: "#141414"
  ink-soft: "#3d4438"
  hair: "#4b5243"
  madder-text: "#8a2232"
  indigo: "#24366b"
  malachite: "#2f6b55"
  umber: "#6b4a2b"
  ochre: "#a97b22"
  madder: "#9e2b3a"
  cobalt-violet: "#5e3f73"
  verdigris: "#2f7d74"
  vermilion: "#b4462c"
typography:
  display:
    fontFamily: "Bodoni Moda, Georgia, serif"
    fontSize: "31px"
    fontWeight: 600
    fontVariation: "'opsz' 31"
  headline:
    fontFamily: "Bodoni Moda, Georgia, serif"
    fontSize: "19px"
    fontWeight: 550
  stamp-word:
    fontFamily: "Bodoni Moda, Georgia, serif"
    fontSize: "18px"
    fontWeight: 650
    letterSpacing: "1.6px"
  figure:
    fontFamily: "Libre Franklin, Segoe UI, sans-serif"
    fontSize: "30px"
    fontWeight: 600
    fontFeature: "'tnum' 1, 'lnum' 1"
  body:
    fontFamily: "Libre Franklin, Segoe UI, sans-serif"
    fontSize: "14px"
    fontWeight: 400
  label:
    fontFamily: "Libre Franklin, Segoe UI, sans-serif"
    fontSize: "13px"
    fontWeight: 500
    fontFeature: "'tnum' 1, 'lnum' 1"
rounded:
  water: "8px"
  lip: "13.5px"
  tray: "16px"
  stamp: "0px"
spacing:
  lip-side: "14px"
  lip-top: "132px"
  lip-foot: "36px"
  window-gutter: "28px"
  rack-gap: "24px"
components:
  tray:
    backgroundColor: "{colors.rim}"
    rounded: "{rounded.tray}"
  tray-water:
    backgroundColor: "{colors.size-wet}"
    rounded: "{rounded.water}"
  pulled-sheet:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
  verdict-stamp-passed:
    textColor: "{colors.ink}"
    rounded: "{rounded.stamp}"
    width: "296px"
    height: "110px"
  verdict-stamp-failed:
    textColor: "{colors.madder-text}"
    rounded: "{rounded.stamp}"
    width: "296px"
    height: "110px"
  view-chip:
    textColor: "{colors.ink-soft}"
    typography: "{typography.body}"
    height: "32px"
  view-chip-active:
    textColor: "{colors.ink}"
  call-out:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
---

# Design System: pinnflow viewer

## Overview

**Creative North Star: "The Marbling Bath"**

The window is a marbler's tray of size. Each scene lays pigment on the water. The trained network's predicted velocity carries that pigment live, and the reference sits beside it as a pulled print on a drying rack. The world is daylight and matte: a grey-green ground, a wet tray with a lighter lip, paper with a torn edge, and ink. There is no dark viewport, no scene tree and no property panel. The ParaView-style viewer is the anti-reference.

Density is low and the image leads. One tray dominates every screen, and its lip carries the labels: the title and summary on the left, the verdict stamp on the right, and how the pigment is moving underneath. Numbers show their source. A measured number is stamped in solid type, and a number derived live from the current frame is set in hairline. Passing and failing scenes get the same care, and every failure shows its own evidence.

**Key Characteristics:**
- One historic marbling pigment per scene, used everywhere that scene appears: drop, strike, playhead, curve and stamp accent.
- Data colour maps are interpolated in OKLab from the pigments, so lightness is monotonic and equal steps look equal.
- Every material is procedural and made at run time: deckled paper, grain, ink density and blurred shadows. No raster file ships.
- Every choice stays on screen. The active one is struck forward by weight, a rule and the scene's pigment.

## Colors

A muted daylight bath with saturated, historic pigments used sparingly.

### Primary
- **The seven pigments** (indigo, malachite, umber, ochre, madder, cobalt violet, verdigris): one per benchmark scene, in benchmark order. **Vermilion** is the eighth, for the real-world use case (the stenosis), the only scene on its tab. A pigment marks its scene: the rail drop, the strike under it, the playhead, the network's curve and the dye in the bath. A pigment is not a text colour. Ochre in particular is only for rules and markers, at 2.3:1 on the ground.

### Neutral
- **Size** (ground): the window ground, the bath's resting water.
- **Wet size** and **deep size**: the water inside the tray, and the tray's walls and edge.
- **Rim**: the tray's lip, which carries all the scene's labels.
- **Paper**: pulled sheets, call-out slips and the stamp's ground.
- **Stone**: obstacles and anything outside the fluid, such as the cylinder and NaN cells.
- **Ink**: primary text and the passed stamp, at 11.3:1 on size.
- **Ink soft**: secondary text, tinted from the size (6.2:1 on size).
- **Hair**: hairline rules and live-derived numbers (5.0:1 on size, 5.6:1 on rim).
- **Madder text**: madder deepened for text and the failed stamp (5.5:1 on size).

### Named Rules
**The Pigment Is the Scene Rule.** A scene's pigment never appears on another scene's controls, and a scene never borrows another's pigment. Dropped pigment cycles through the drop order and is the only exception.

**The Data Colour Rule.** Field views use the OKLab maps only: sequential (pale size, verdigris, indigo, deep ink), diverging (madder, pale size, indigo) and error (pale, ochre, madder, lake). Decorative pigment never encodes a value.

## Typography

**Display Font:** Bodoni Moda, a variable font with optical size and weight axes (Georgia fallback).
**Body Font:** Libre Franklin, a variable weight font with tabular and lining figures (Segoe UI fallback).

**Character:** A high-contrast Didone names things, the way marbling plates were titled. A sturdy grotesque with tabular figures carries every number and control.

### Hierarchy
- **Display** (Bodoni 600, 31 px, optical size 31): the scene title on the tray's lip, one line only.
- **Headline** (Bodoni 550, 19 px): the title of a pulled sheet.
- **Stamp word** (Bodoni 650, 18 px, capitals, tracked 1.6 px): the verdict word.
- **Figure** (Franklin 600, 30 px, tabular): the measured number on the stamp and the recovered coefficients.
- **Body** (Franklin 400, 14 px): summaries, captions and the empty-sheet notes.
- **Label** (Franklin 500, 13 px, tabular): ticks, legends, verdict words, the mode line and the live line.

### Named Rules
**The 13 Pixel Floor.** No text in the window is smaller than 13 px. The audience reads it on a projector across a room.

**The Tabular Number Rule.** Every number is set with tabular, lining figures. Small values use one scientific notation, with no zero-padded exponent (4.2e-5). A live percentage takes the measured headline's precision.

## Layout

The window has three bands:
- **A rail of seven drops** across the top.
- **A body** holding the tray and the drying rack.
- **A bottom rim** with play, the time or Reynolds bar, the view chips, the device timing line and the hint.

The tray sizes itself from the domain's aspect ratio, so the water hugs the flow. It is never narrower than its lip needs: the title on one line, at most four lines of summary, and the stamp. It takes at most 70% of the width, and the rack takes the rest, with a 24 px gap. A domain wider than 2.2:1, such as the cylinder channel, gets a full-width tray. The rack then becomes a row beneath it: a note card, the chart, and a 360 px caption column. In the column layout, the rack stacks the pulled sheet over the chart over the caption. A scene without a reference field shrinks its sheet to a note card, and the chart takes the room. The window gutter is 28 px, and the minimum window is 1240 by 760.

## Elevation & Depth

Depth is physical and low. Objects that sit on the ground cast one soft shadow: the tray, the pulled sheets and the stamp's impression. Shadows are blurred, offset downwards, and in ink at 20 to 22% opacity. Inside the tray, depth is tonal: the deep wall, the lighter lip and a fine paper-coloured highlight on the lip's inner edge.

### Shadow Vocabulary
- **Tray shadow** (gaussian blur 6 px, 4 px down, ink at 22%): under the tray's rounded rectangle.
- **Sheet shadow** (gaussian blur 6 px, 4 px down and 2 px right, ink at 20%): follows the paper's deckled alpha.

### Named Rules
**The No Stacked Offsets Rule.** A shadow is one blurred mask, never a stack of zero-blur offset copies.

## Shapes

The bath is soft: the tray (16 px), the lip (13.5 px) and the water (8 px) are all rounded. Paper is torn: sheets have a fibrous deckled edge of a few pixels' wander, made from smoothed noise. Ink is square: the verdict stamp has square corners and a mitred double rule (2.4 px outside, 0.9 px inside, 4.5 px apart), set at -1.5 degrees. Drops are organic blots with seven soft lobes, used for the rail, the playheads and the Reynolds marker.

## Components

### Verdict stamp
An impression, not a box. It is painted offscreen in the verdict ink: ink for passed and gate missed, madder text for failed. It carries the word in Bodoni capitals at the top left, and the measured number at the top right over a hairline. Below are a label of two balanced lines, in which no unit ever starts a line, and "measured, STATUS.md". A procedural ink-density mask is applied with destination-in compositing. On a scene change, the impression presses in over 220 ms, scaling from 1.05 to 1 and fading from 25% to full.

### Tray
A lip in rim colour with the deep wall inside it and the wet water within that. The top lip holds the title block and the stamp. The bottom lip holds the mode line on the left and the live line on the right. The 2D bath and the 3D tank both sit in the same tray.

### Pulled sheet
Procedural paper with a deckled alpha edge, faint fibre grain at 2.8% and its own blurred shadow. It carries a Bodoni title, the reference printed at the bath's colour scale with a hairline frame, and a 13 px note. The empty state is a 14 px note in ink soft.

### Call-out
A measured value pinned to a place in the flow. It has a 9 px ink ring with a pale halo, an elbow leader into the water, and a paper slip with ink text at 13 px tabular. It is used on both cavities' corners in the divergence view, and both lids get one so the comparison is fair.

### Tab strip
Two tabs over the rail: "The seven benchmarks" and "Real-world use case". The text is Franklin at 15 px; the active tab is ink at weight 650, with a 2 px ink underline that slides between tabs over 200 ms. T toggles the tab. The rail keeps one cell width (a seventh of the window) on every tab, so a drop never moves. The real-world tab's single drop is followed by a 14 px note saying it is a synthetic study, not a clinical tool.

### Rail, chips and bars
- **Rail:** seven drops with short names and verdict words. The active drop grows, gains an ink outline and is underlined by a strike in its pigment. The strike slides between cells over 200 ms with an exponential ease-out. An unfilled drop means that scene is still compiling on the GPU.
- **View chips:** text at a fixed active-weight width, so nothing shifts. A pigment underline slides to the chosen chip. A madder dot marks the view that holds a failure's evidence.
- **Time bar:** the whole range is on screen, with a pigment fill and a blot playhead. It can show another unit: the heartbeat in seconds. The Reynolds bar is either discrete marks or a log scale with the held-out band hatched.
- **Easing:** playheads ease only on real jumps, over 4% of the range, never during playback.
- **Focus:** every control takes Tab focus and arrow keys. The focus ring is a 1.5 px ink rule shown only after keyboard focus.

### Coefficient panel
Recovered coefficients in ink, at 30 px with tabular figures, every value's decimal point on one x. Each sits beside its true value and its relative error. A small ochre dot marks each row.

## Do's and Don'ts

### Do:
- **Do** put a scene's labels on the tray's lip, not on the ground above it.
- **Do** show every measured number with its source, and every live number in hairline.
- **Do** drop the pigment to a 20% veil over a data view, so the evidence reads first.
- **Do** give a failing scene the same care as a passing one, with its specific evidence on screen: the corner divergence, the lost flow rate or the missing dissipation peak.
- **Do** make materials procedurally, cached per pixel size.

### Don't:
- **Don't** use a dark viewport, a scene tree or a property panel.
- **Don't** set text in a pigment. Ochre especially fails contrast; it is a rule and marker colour.
- **Don't** draw a missing value as zero. Leave the station or cell out, or paint it stone.
- **Don't** claim the networks are faster or cheaper than a CFD solver. The window shows accuracy and live inference, nothing more.
