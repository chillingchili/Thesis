---
name: Pickleball Serve Coach
description: An on-device serve coach that measures, labels its uncertainty, and shows every result against its reference.
colors:
  accent: "#262626"
  on-accent: "#FFFFFF"
  accent-muted: "#F1F1F1"
  body: "#F1F1F1"
  surface: "#FFFFFF"
  menu-surface: "#FFFFFF"
  menu-stroke: "#D4D4D4"
  text-primary: "#171717"
  text-secondary: "#525252"
  text-disabled: "#A3A3A3"
  icon-secondary: "#737373"
  border: "#00000014"
  border-emphasis: "#D4D4D4"
  ok: "#007004"
  ok-muted: "#C5E5C0"
  warn: "#745B00"
  warn-muted: "#F8DA9D"
  err: "#A50C25"
  err-muted: "#FACECB"
  select: "#0074E2"
  accent-dark: "#EBEBEB"
  on-accent-dark: "#171717"
  accent-muted-dark: "#333333"
  body-dark: "#1B1B1B"
  surface-dark: "#262626"
  menu-surface-dark: "#333333"
  menu-stroke-dark: "#FFFFFF1A"
  text-primary-dark: "#FAFAFA"
  text-secondary-dark: "#A3A3A3"
  text-disabled-dark: "#525252"
  icon-secondary-dark: "#A3A3A3"
  border-dark: "#FFFFFF1A"
  border-emphasis-dark: "#525252"
  ok-dark: "#9FE59B"
  ok-muted-dark: "#84C9803D"
  warn-dark: "#FDCF4F"
  warn-muted-dark: "#DEB4333D"
  err-dark: "#FFC6C1"
  err-muted-dark: "#FF9E973D"
  select-dark: "#6D9CFE"
  camera-bg: "#000000"
  chip-scrim: "#171717DB"
  on-camera: "#FAFAFA"
  on-camera-secondary: "#A3A3A3"
  on-camera-chip: "#FFFFFF"
  on-camera-chip-text: "#171717"
  framing-ok: "#3DDC84"
  framing-keyline: "#0000008C"
  dot-ok: "#64AF4C"
  dot-warn: "#FFCE2F"
  dot-err: "#FF705D"
typography:
  cue:
    fontFamily: "Figtree, sans-serif"
    fontSize: "24px"
    fontWeight: 600
    letterSpacing: "-0.01em"
  dialog-title:
    fontFamily: "Figtree, sans-serif"
    fontSize: "20px"
    fontWeight: 600
  value:
    fontFamily: "Figtree, sans-serif"
    fontSize: "22px"
    fontWeight: 600
    fontFeature: "tnum"
  title:
    fontFamily: "Figtree, sans-serif"
    fontSize: "15px"
    fontWeight: 600
  label:
    fontFamily: "Figtree, sans-serif"
    fontSize: "15px"
    fontWeight: 500
  body:
    fontFamily: "Figtree, sans-serif"
    fontSize: "15px"
    fontWeight: 400
  button:
    fontFamily: "Figtree, sans-serif"
    fontSize: "15px"
    fontWeight: 600
  badge:
    fontFamily: "Figtree, sans-serif"
    fontSize: "14px"
    fontWeight: 500
  reference:
    fontFamily: "Figtree, sans-serif"
    fontSize: "14px"
    fontWeight: 500
    fontFeature: "tnum"
  small:
    fontFamily: "Figtree, sans-serif"
    fontSize: "13px"
    fontWeight: 400
rounded:
  sm: "6px"
  md: "10px"
  lg: "12px"
  dialog: "28px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "10px"
  lg: "12px"
  panel-inset: "14px"
  touch: "48px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.on-accent}"
    typography: "{typography.button}"
    rounded: "{rounded.md}"
    height: "48px"
  button-outlined:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text-primary}"
    typography: "{typography.button}"
    rounded: "{rounded.md}"
    height: "48px"
  button-text:
    textColor: "{colors.text-primary}"
    typography: "{typography.button}"
    rounded: "{rounded.md}"
    height: "48px"
  button-icon:
    textColor: "{colors.text-primary}"
    rounded: "{rounded.md}"
    size: "48px"
  badge-good:
    backgroundColor: "{colors.ok-muted}"
    textColor: "{colors.ok}"
    typography: "{typography.badge}"
    rounded: "{rounded.sm}"
    padding: "3px 8px"
  badge-warn:
    backgroundColor: "{colors.warn-muted}"
    textColor: "{colors.warn}"
    typography: "{typography.badge}"
    rounded: "{rounded.sm}"
    padding: "3px 8px"
  badge-neutral:
    backgroundColor: "{colors.accent-muted}"
    textColor: "{colors.text-secondary}"
    typography: "{typography.badge}"
    rounded: "{rounded.sm}"
    padding: "3px 8px"
  panel:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.lg}"
    padding: "0 8px 8px 14px"
  status-chip:
    backgroundColor: "{colors.chip-scrim}"
    textColor: "{colors.on-camera}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    padding: "6px 12px 6px 10px"
  segment:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text-secondary}"
    typography: "{typography.badge}"
    rounded: "{rounded.md}"
    height: "44px"
    padding: "0 14px"
  menu:
    backgroundColor: "{colors.menu-surface}"
    textColor: "{colors.text-primary}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
  framing-hint:
    backgroundColor: "{colors.on-camera-chip}"
    textColor: "{colors.on-camera-chip-text}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    padding: "5px 12px"
  segment-checked:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.on-accent}"
  serve-chip:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text-primary}"
    typography: "{typography.badge}"
    rounded: "{rounded.md}"
    height: "48px"
    padding: "0 12px"
  fact-block:
    backgroundColor: "{colors.accent-muted}"
    textColor: "{colors.text-primary}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
---

# Design System: Pickleball Serve Coach

## Overview

**Creative North Star: "The Instrument Beside the Court"**

The app is one quiet, neutral instrument standing next to a live camera. It is built from Meta's Astryx Neutral 0.4.1 tokens applied through stock Material 3 components on Android Views, so every control behaves like the platform and every surface reads as the same grey-and-white paper. Nothing on the panel is decorative: the camera supplies all the imagery, and the panel supplies words, numbers, and the reference each number is judged against.

Density is set for a glance from two to three metres in landscape. The coaching cue leads at the largest size; three measured rows follow, each with its label, a help button, and its reference on one line, then the value and a worded badge on the next. Color is reserved for status (green, amber, red) and selection (blue); everything else is the neutral ramp from near-black to near-white, in both light and dark themes.

The system is honest by construction. Status is always a word with a tint, never a tint alone; a result below the model's confidence gate is shown and labeled uncertain rather than hidden or rounded up; and every measured value sits beside the band it is compared with.

**Key Characteristics:**
- Astryx Neutral tokens through native Material 3 roles, light and dark.
- One surface panel beside a true 4:3 camera; the camera never moves.
- Figtree throughout; three sizes carry the live screen (cue 24, value 22, label 15).
- Flat: 1dp hairline borders and tonal steps, no resting shadows.
- Lucide icons at 2dp stroke, tinted by token.
- Worded status badges in three tones; uncertainty always labeled.

## Colors

A neutral grey ramp carries the whole interface; hue appears only to report status or mark the current selection.

### Primary
- **Graphite Ink** (accent): filled buttons, the checked segment, probability bar fill, and the Material primary role. In dark theme it inverts to **Pale Ash** (accent-dark) with near-black text on it. It is deliberately not a brand hue: the primary action is the darkest neutral.
- **Fog Step** (accent-muted): secondary-container fills, the neutral badge, fact blocks in dialogs, the probability track, and the highest surface container.

### Tertiary
- **Selection Blue** (select): the 2dp stroke around the selected serve chip in review, and Material's tertiary role. It means "this one is selected" and nothing else.

### Status
- **Court Green** (ok) on **Mint Wash** (ok-muted): "Within reference", "Passed", "Coach rules", and the setup check that has passed.
- **Amber Ochre** (warn) on **Straw Wash** (warn-muted): "Needs work", "Outside reference", "Low confidence", "Below gate", "General policy · unrated"; also the value text of an uncertain serve type, uncertain serve chips in review, and the 0.60 gate tick on probability bars.
- **Signal Red** (err) on **Blush Wash** (err-muted): mapped to Material's error role for errors; no live result uses it.
- In dark theme the status tones lighten (ok-dark, warn-dark, err-dark) and their washes become 24% translucent tints of the same hue.

### Neutral
- **Body Grey** (body): window and page background behind the camera and panel.
- **Paper White** (surface): the panel, cards, outlined controls, serve chips. Every Material surface-container step except the highest maps here, so surfaces do not drift into tinted greys.
- **Ink** (text-primary), **Graphite** (text-secondary), **Ash** (text-disabled): primary text and values; labels, references, tips, and unchecked segment text; disabled text.
- **Slate Icon** (icon-secondary): help buttons and pending setup checks.
- **Hairline** (border, 8% black / 10% white in dark): panel border, row dividers, serve-chip stroke.
- **Rule Grey** (border-emphasis): outlined button and segment strokes, Material outline.
- **Menu Surface** (menu-surface) and **Menu Stroke** (menu-stroke): the overflow popup. In light it is paper white with a Rule Grey edge; in dark it steps up to #333333 with a hairline edge.

### On-camera (theme-invariant)
- **Camera Black** (camera-bg), **Chip Scrim** (chip-scrim, 86% ink), **Camera White** (on-camera), **Camera Grey** (on-camera-secondary): the status chip and replay controls that sit on video.
- **Camera Chip White** (on-camera-chip) with **Camera Chip Ink** (on-camera-chip-text): the light chips that sit on video, the framing hint ("Stand inside the outline") and the retry button.
- **Framing Green** (framing-ok) over **Framing Keyline** (framing-keyline, 55% black): the player outline. A 4.5dp keyline is drawn under a 2.5dp stroke so the outline reads on any background; the stroke is dashed white until framing passes, then solid Framing Green.
- **Status dots** (dot-ok, dot-warn, dot-err): the 8dp dot in the status chip. Dot-warn also draws the pose skeleton and peak marker on replayed clips.

### Named Rules
**The Status-Only Hue Rule.** Green, amber, and red appear only to report a measured status, and blue only to mark selection. A neutral control never takes a status hue for emphasis.

**The Words With Color Rule.** Every status tint carries a word: a badge label, a check-row sentence, or a status-chip line. A screen read in greyscale loses no result.

**The Fixed Camera Rule.** Anything drawn on the camera image uses the on-camera tokens, which are identical in light and dark themes, because the video does not change with the theme.

## Typography

**Display Font:** Figtree (bundled, 400/500/600/700)
**Body Font:** Figtree
**Label/Mono Font:** Figtree with tabular figures (`tnum`) for every number

**Character:** One geometric, friendly sans at four weights. Hierarchy comes from size and weight steps, never from a second family, uppercase, or tracking.

### Hierarchy
- **Cue** (600, 24sp, -0.01em): the coaching sentence that leads the live panel. Live shows the first cue only, as a plain sentence; Read aloud and the review keep every cue. If the cue would push a metric row off the panel, the screen gives way in a fixed order: hide the notes line, step the size down to a 20sp floor, drop the leading diagnosis sentence, then clamp to 3 lines.
- **Dialog Title** (600, 20sp): Material dialog titles and the headline/title-large roles.
- **Value** (600, 22sp, tabular): the measured result on each live row ("Limited", "11°", "Not sure").
- **Title** (600, 15sp): Material title-medium; the value column in the research table.
- **Label** (500, 15sp, text-secondary): row labels, the setup title, check rows (in text-primary), status-chip line.
- **Reference** (500, 14sp, tabular, text-secondary): the reference band at the right end of each row head.
- **Body** (400, 15sp): dialog bodies, waiting text; tips and notes at 14sp in text-secondary.
- **Badge** (500, 14sp): status badges, segments, serve chips.
- **Button** (600, 15sp, sentence case, no tracking): every button label.
- **Small** (400, 13sp, text-secondary): table headers, notes under the cue, status-chip detail.

### Named Rules
**The Three Sizes Rule.** The live screen is carried by three sizes: cue 24, value 22, label 15. A new live element takes one of them before it takes a new size.

**The Rows Before Words Rule.** On the live screen the metric rows always stay visible. A long cue gives up its notes, then size (down to 20sp, never lower), then its opening diagnosis, then lines (3 maximum), never a row.

**The Tabular Numbers Rule.** Every number that can change between serves (values, references, probabilities, timecodes) sets `tnum`, so columns and rows do not jitter.

**The Sentence Case Rule.** Buttons and labels are sentence case with zero letter-spacing; nothing in the system is uppercase.

## Layout

Landscape, two columns inside an 8dp window gutter. The camera frame sits left at full height, sized to a true 4:3 aspect by `AspectFrameLayout`, with the preview in fitCenter so the picture is exactly the frame the model analyzes. The panel takes the remaining width after an 8dp gap. The camera never moves between Setup and Live; only the panel's contents switch.

Inside the panel: 14dp start inset, 8dp end and bottom. A 48dp bar holds the screen's way out or way in (setup title, or "All serves · N ›"), the Read aloud switch, and the overflow menu. Below it, Live mode stacks the cue, an optional neutral badge, notes, then the metric rows separated by 1dp dividers. Each metric row is a 28dp head (label, 48dp help target pulled into the line, reference right-aligned) over a value row of at least 28dp (value plus badge 10dp after it), with 6dp below. Setup stacks 36dp check rows (20dp icon, 10dp gap), then 28dp tips with a 10×2dp dash, then the start note and a full-width 48dp button at the bottom.

The review screen uses the same shell: clip player left on camera black, panel right with a back arrow, serve title, and a Result | Research segmented control; serve chips run in a strip under the clip. The Result tab shows the first cue, then a "Show all cues (N)" text button. Expanded, the cues are a bulleted list with a hanging indent, so wrapped lines align under the text, not under the bullet.

Spacing steps observed: 4, 8, 10, 12, 14dp; touch targets are 48dp (segments 44dp, check rows 36dp of text rows, not targets).

**The Reference On Every Row Rule.** A measured value never appears without the reference it is judged against on the same row. Live rows read "needs ≥ 60%", "0.03 (≥ 0.154)" and "band 13.4–165.7°"; the research table has a Reference column. Thresholds read the same everywhere (≥ 0.60, ≥ 0.154, 13.4–165.7°). A new metric ships with its reference or does not ship.

## Elevation & Depth

Flat. Surfaces have no resting shadows. Depth comes from the tonal step between body and surface, plus a 1dp border around the panel, the menu, and the serve chips. The overflow menu has its own border color (menu-stroke): Rule Grey in light, the hairline in dark, where the menu also sits one tonal step above the panel. On the camera, the framing outline gets its depth from its translucent keyline, not a shadow. On the camera, depth comes from scrims: the 86% ink status chip, and black-to-transparent gradients behind replay controls (75% at the bottom) and the top bar (72% at the top). Material components keep their own platform elevation behavior (dialogs, snackbars, switch thumb); the system adds none.

**The Flat Panel Rule.** The panel and everything on it are flat. Separation is a hairline or a tonal step, never a shadow.

**The Dim Not Hide Rule.** While the next serve is analyzed, the last result stays on screen at 45% opacity rather than disappearing.

## Shapes

Gently rounded, on a three-step scale mapped to Material's corner roles: small 6dp (badges, status chip, framing hint), medium 10dp (buttons, segments, serve chips, menu, fact blocks, snackbar), large 12dp (panel, camera frame). Dialogs keep Material's 28dp. The camera frame clips its video to the 12dp corner so the image and panel share a silhouette. Strokes are 1dp; the selected serve chip takes a 2dp selection stroke. Probability bars are 8dp tracks with 4dp ends. The gate is marked by a drawn 2dp amber tick: 2×16dp across each bar, and a 2×12dp swatch next to the "0.60 gate" legend.

## Components

### Buttons
Platform Material buttons, flattened and enlarged for a court.
- **Shape:** medium corners (10dp), no vertical insets, at least 48dp tall.
- **Primary (filled):** Graphite Ink fill with white label ("Try again"); used once per state at most.
- **Outlined:** surface with a Rule Grey 1dp stroke and ink label ("Start now", "Setup tips", "Setup"); also the dialog positive button.
- **Text:** ink label, no container ("All serves · N ›" with a trailing 18dp chevron); dialog negative and neutral buttons.
- **Icon:** 48dp square target, 20dp Lucide icon in ink (menu). The help variant is 18dp in Slate Icon and is pulled into the row head with -10dp vertical margins.

### Status Badges
The worded verdict next to every value.
- **Style:** small corners (6dp), 8dp by 3dp padding, Badge type, no border. Good: Court Green on Mint Wash. Warn: Amber Ochre on Straw Wash. Neutral: Graphite on Fog Step (with a hairline when used as the cue badge).
- **Vocabulary (live):** "Within reference", "Needs work", "Outside reference", "Too open", "Too closed", "Low confidence", "Partial". **(research):** "Passed", "Below gate", "Pass", "Below", "Within", "Outside", "Not measured"; cue source "Coach rules" or "General policy · unrated".

### Metric Row (signature)
The unit of the live screen and the review Result tab: label + help + reference on top, value + badge below, hairline between rows. An uncertain serve type shows "Not sure" with the value itself in Amber Ochre and a "Low confidence" badge; a measurement that could not be taken shows "Not measured" or "Not detected" with a "Partial" badge.

### Status Chip (on camera)
Top-left of the camera at 10dp: Chip Scrim fill and small corners. It holds an 8dp status dot, or a Material CircularProgressIndicator while busy (14dp, 2dp track, Camera White), then one Label line in Camera White, max 360dp. A second Small line in Camera Grey appears only for problems.

### Framing Hint and Outline (on camera)
The hint is a Camera Chip White chip with small corners and Label type in Camera Chip Ink, centred 12dp above the bottom edge of the camera. The outline is the arch the player stands in; see Framing Green under Colors.

### Setup Checks
Rows the app measures itself: a 20dp Lucide circle in Slate Icon while pending, check-circle in Court Green when passed, info-outline in Amber Ochre when it needs attention, each followed by a sentence in ink.

### Segmented Control
Result | Research in review: outlined, 44dp, medium corners, Rule Grey stroke; unchecked is transparent with Graphite text, checked fills Graphite Ink with on-accent text.

### Serve Chips
Review strip of past serves ("3 Not sure · 1:32"): surface fill, hairline stroke, medium corners, 48dp. Selected takes a 2dp Selection Blue stroke; an uncertain serve's text turns Amber Ochre.

### Navigation
There is no app bar or nav bar. The panel bar carries navigation: a text button into the review, a back arrow out of it, and an overflow menu behind a Lucide ellipsis-vertical icon. The menu is an androidx PopupMenu styled by Widget.Serve.PopupMenu: Menu Surface, a 1dp Menu Stroke edge, medium corners (10dp), offset 8dp in from the edge and 4dp down. Research tools are marked with the Lucide bar-chart icon. Read aloud is a Material switch with a Label-type caption.

### Dialogs and Help
Material alert dialogs at 28dp corners with Dialog Title and Body type; help content groups facts in Fog Step blocks with medium corners.

### Icons
Lucide (ISC), 24-unit viewBox, 2dp stroke, round caps and joins, no fill, tinted by token: ink for actions, Slate Icon for help and pending, status colors only on setup checks.

## Do's and Don'ts

### Do:
- **Do** put the reference on every row that shows a measured value, in Reference type at the row's right end.
- **Do** pair every status color with a word: badge text, a check sentence, or a status-chip line.
- **Do** show uncertain results and label them: "Not sure" in Amber Ochre with "Low confidence", or "General cue · serve type uncertain" under the cue.
- **Do** map new colors to Astryx tokens through Material roles in colors.xml and values-night, so light and dark stay paired.
- **Do** use on-camera tokens for anything drawn over video, unchanged across themes, and give thin strokes on video a keyline.
- **Do** show only the first cue live, and keep the full list behind "Show all cues (N)" in review.
- **Do** keep touch targets at 48dp and numbers in tabular figures.
- **Do** draw new icons from Lucide at 2dp stroke and tint them by token.

### Don't:
- **Don't** convey a result by color alone, or use green, amber, or red as decoration.
- **Don't** hide or round up a result below the 0.60 confidence gate.
- **Don't** add resting shadows to the panel, rows, chips, or menu; separate with a hairline or a tonal step.
- **Don't** split the live panel into equal cards or add decorative sports chrome; one surface carries the cue and the rows.
- **Don't** add a second typeface, uppercase labels, or letter-spaced labels.
- **Don't** hard-code hex values in layouts or views where a token exists.
