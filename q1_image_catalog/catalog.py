"""Q1: image catalog + auto detection + manual correction + composite generation.

Laziest stack that covers the brief: Pillow + numpy + scipy, no model weights,
no API keys. Detection = background removal + connected components + color/shape
heuristics. Works on cartoon AND photo-style assets because the catalog uses
plain backgrounds; anything it gets wrong is fixed by manual correction, which
is the source of truth until detection is explicitly rerun.
"""
import os
import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label as paint_label

BG_THRESH = 235       # channels above this count as (near-)white background
MIN_AREA = 150        # smaller regions are anti-aliasing noise
SPLIT_SHARE = 0.15    # color share that counts as a "major" color

PALETTE = {  # name -> rgb; auto color naming snaps to the nearest entry
    "red": (230, 30, 30), "orange": (245, 130, 20), "yellow": (240, 210, 30),
    "green": (40, 160, 60), "blue": (40, 100, 230), "purple": (140, 60, 200),
    "pink": (240, 150, 200), "brown": (150, 90, 40), "skin": (255, 214, 170),
    "black": (30, 30, 30), "gray": (150, 150, 150), "navy": (40, 40, 90),
}
AUTO_FIELDS = ("bbox", "color", "colors", "confidence", "shape", "label")
USER_FIELDS = ("name", "color", "bbox", "confidence")  # user-editable subset


def color_name(rgb):
    rgb = np.asarray(rgb, dtype=float)
    return min(PALETTE, key=lambda n: sum((a - b) ** 2 for a, b in zip(PALETTE[n], rgb)))


def _shape_of(area, bbox):
    x, y, w, h = bbox
    fill = area / max(1, w * h)
    squarish = abs(w - h) <= max(4, 0.12 * max(w, h))
    if fill > 0.88:
        return "square" if squarish else "rectangle"
    if (w > 1.6 * h or h > 1.6 * w) and fill < 0.9:
        return "figure"  # tall/wide articulated thing (person), not a shape
    if fill > 0.62:
        return "circle"
    if fill > 0.38:
        return "triangle"
    return "figure"


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix, iy = max(ax, bx), max(ay, by)
    iw, ih = min(ax + aw, bx + bw) - ix, min(ay + ah, by + bh) - iy
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter)


def detect_objects(image):
    """Return [dict(label,color,colors,bbox,confidence,shape)] for a PIL image."""
    arr = np.asarray(image.convert("RGB"))
    H, W, _ = arr.shape
    fg = ~(arr > BG_THRESH).all(axis=2)
    if fg.sum() < max(MIN_AREA, 0.005 * H * W):
        return []  # no detectable objects
    out, cands = [], []
    # One foreground component = one object (touching parts, e.g. a cartoon
    # figure, stay merged; separated items stay separate).
    lab, n = paint_label(fg, structure=np.ones((3, 3)))
    for i in range(1, n + 1):
        m = lab == i
        if int(m.sum()) >= MIN_AREA:
            cands.append(m)
    biggest = max(int(m.sum()) for m in cands)
    for m in cands:
        area = int(m.sum())
        if area < max(MIN_AREA, 0.15 * biggest):
            continue  # satellite speck (leaf, stem): part of the main object
        ys, xs = np.where(m)
        box = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
               int(ys.max() - ys.min() + 1))
        mean = arr[m].mean(axis=0)
        # Major colors inside this object (for clothing palettes / overlaps).
        # Regions are per-color *connected* components, so disjoint same-color
        # parts (head vs legs) get their own boxes instead of one giant one.
        q = (arr // 64 * 64)
        uniq, counts = np.unique(q[m].reshape(-1, 3), axis=0, return_counts=True)
        majors = []
        for c, k in sorted(zip(uniq, counts), key=lambda t: -t[1]):
            if k < SPLIT_SHARE * area or (c > BG_THRESH).all():
                continue
            sm = m & (q == c).all(axis=2)
            rl, rn = paint_label(sm, structure=np.ones((3, 3)))
            for r in range(1, rn + 1):
                rm = rl == r
                if int(rm.sum()) < MIN_AREA:
                    continue
                sy, sx = np.where(rm)
                rb = (int(sx.min()), int(sy.min()), int(sx.max() - sx.min() + 1),
                      int(sy.max() - sy.min() + 1))
                majors.append((color_name(arr[rm].mean(axis=0)), rb, rm))
        if len(majors) >= 2:
            # Occlusion (split) vs adjacency/detail (merge): split only if two
            # regions overlap substantially yet neither contains the other.
            # A leaf inside an apple box is ~100% contained -> detail, merged.
            boxes = [b for _, b, _ in majors]
            split = False
            for k in range(len(boxes)):
                for l in range(k + 1, len(boxes)):
                    a, b = boxes[k], boxes[l]
                    iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
                    ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
                    inter = max(0, iw) * max(0, ih)
                    small = min(a[2] * a[3], b[2] * b[3])
                    if inter > 0.005 * H * W and inter / small < 0.7:
                        split = True
            if split:  # truly overlapping -> one object per color
                bycol = {}
                for col, rb, rm in majors:
                    bycol.setdefault(col, []).append((rb, rm))
                for col, parts in bycol.items():
                    xs0 = min(p[0] for p, _ in parts)
                    ys0 = min(p[1] for p, _ in parts)
                    xs1 = max(p[0] + p[2] for p, _ in parts)
                    ys1 = max(p[1] + p[3] for p, _ in parts)
                    sb = (xs0, ys0, xs1 - xs0, ys1 - ys0)
                    shape = _shape_of(sum(int(x.sum()) for _, x in parts), sb)
                    out.append({"label": f"{col} {shape}", "color": col,
                                "colors": [col], "bbox": sb, "confidence": 0.55,
                                "shape": shape})
                continue
        cols = [c for c, _, _ in majors] or [color_name(mean)]
        shape = _shape_of(area, box)
        solid = area / (box[2] * box[3])
        conf = round(min(0.95, 0.45 + 0.45 * solid + 0.05 * (area / fg.sum())), 2)
        out.append({"label": f"{cols[0]} {shape}", "color": cols[0],
                    "colors": cols, "bbox": box, "confidence": conf, "shape": shape})
    return out


@dataclass
class DetObj:
    id: int
    name: str                      # USER-EDITABLE (defaults to auto label)
    auto_label: str                # auto snapshot, never user-edited
    color: str
    colors: list = field(default_factory=list)
    bbox: tuple = (0, 0, 0, 0)
    confidence: float = 0.0
    shape: str = ""
    corrected: bool = False        # True once a human touches it
    auto_snapshot: dict = field(default_factory=dict)


@dataclass
class Entry:
    name: str
    image: Image.Image
    objects: list = field(default_factory=list)
    status: str = "detected"       # detected|no_objects|manual|composite
    source: dict = field(default_factory=dict)  # composite provenance


class Catalog:
    """Reusable image collection. Manual edits win until rerun_detection()."""

    def __init__(self):
        self.items = {}
        self._next_id = 1

    def add_image(self, name, image, object_names=None, source=None):
        det = detect_objects(image)
        objs = []
        for j, d in enumerate(det):
            uname = (object_names or {}).get(j, d["label"]) if isinstance(
                object_names, dict) else (object_names[j] if object_names and j < len(object_names) else d["label"])
            objs.append(DetObj(self._next_id + j, uname, d["label"], d["color"],
                               d["colors"], d["bbox"], d["confidence"], d["shape"],
                               auto_snapshot={k: d[k] for k in AUTO_FIELDS}))
        self._next_id += len(objs)
        e = Entry(name, image.convert("RGB"), objs,
                  "detected" if objs else "no_objects", source or {})
        self.items[name] = e
        return e

    def _obj(self, entry, oid):
        for o in self.items[entry].objects:
            if o.id == oid:
                return o
        raise KeyError(f"object {oid} not in {entry}")

    def add_object(self, entry, obj_name, bbox, color="gray", confidence=1.0):
        e = self.items[entry]  # manual add when detection missed something
        o = DetObj(self._next_id, obj_name, obj_name + " (manual)", color,
                   [color], tuple(bbox), confidence, "manual",
                   corrected=True, auto_snapshot={})
        self._next_id += 1
        e.objects.append(o)
        e.status = "manual"
        return o

    def edit_object(self, entry, oid, **kw):
        o = self._obj(entry, oid)  # only USER_FIELDS are writable
        bad = set(kw) - set(USER_FIELDS)
        if bad:
            raise ValueError(f"auto-only fields, not editable: {sorted(bad)}")
        for k, v in kw.items():
            setattr(o, k, tuple(v) if k == "bbox" else v)
        o.corrected = True
        self.items[entry].status = "manual"
        return o

    def remove_object(self, entry, oid):
        e = self.items[entry]
        e.objects = [o for o in e.objects if o.id != oid]
        e.status = "manual"

    def rerun_detection(self, entry):
        e = self.items[entry]  # explicit rerun discards manual corrections
        fresh = self.add_image(entry + "__tmp", e.image)
        del self.items[entry + "__tmp"]
        e.objects = fresh.objects
        e.status = "detected" if e.objects else "no_objects"
        return e

    # ---- composite generation: natural language -> reuse of catalog assets ----
    def generate_composite(self, prompt, new_name, canvas_bg="white"):
        wants = self._parse(prompt)
        missing = self._unknown_nps(prompt)
        if not wants and not missing:
            raise ValueError(f"no catalog assets mentioned in: {prompt!r}")
        assets = []
        for count, color, noun in wants:
            hit = [n for n in self.items
                   if n == noun or noun in n or n in noun] or ([noun] if noun in self.items else [])
            if color and not any(o.color == color for n in hit for o in self.items[n].objects):
                missing.append(f"{color} {noun}")  # asked color we don't have
                continue
            if color:
                hit = [n for n in hit if any(
                    o.color == color for o in self.items[n].objects)] or hit
            (assets.extend([hit[0]] * count) if hit else missing.append(noun))
        if missing:  # never silently swap in a generated approximation
            raise ValueError(f"missing catalog assets (add them first): {missing}")
        cell, per = 250, 300
        canvas = Image.new("RGB", (cell * len(assets), per), canvas_bg)
        transforms = []
        for k, a in enumerate(assets):
            im = self.items[a].image
            fg = ~(np.asarray(im.convert("RGB")) > BG_THRESH).all(axis=2)
            alpha = Image.fromarray((fg * 255).astype("uint8"))
            im = im.copy()
            im.thumbnail((cell - 30, per - 30))
            x = k * cell + (cell - im.width) // 2
            y = (per - im.height) // 2
            canvas.paste(im, (x, y), alpha.resize(im.size))
            transforms.append({"asset": a, "pos": (x, y), "size": im.size})
        e = self.add_image(new_name, canvas, source={"prompt": prompt,
                           "sources": assets, "transforms": transforms})
        e.status = "composite"
        return e

    def _parse(self, prompt):
        """Extract (count, color, catalog-noun) triples from free text."""
        out, seen = [], set()
        for noun in sorted(self.items, key=len, reverse=True):
            for m in re.finditer(
                    r"(?:\b(one|two|three|\d+|a|an)\b\s+)?(?:\b(\w+)\b\s+)?\b"
                    + re.escape(noun) + r"s?\b", prompt, re.I):
                if m.span() in seen:
                    continue
                seen.add(m.span())
                cw, adj = (m.group(1) or "one").lower(), (m.group(2) or "").lower()
                n = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3}.get(cw)
                out.append((m.start(), int(cw) if n is None else n,
                            adj if adj in PALETTE else None, noun))
        return [(c, col, n) for _, c, col, n in sorted(out)]

    def _unknown_nps(self, prompt):
        """Noun phrases after a count-word that match no catalog entry.

        Catches requests like 'one dragon fruit': the parser can only resolve
        known nouns, so anything else that looks like an asset request must be
        reported, never silently dropped or AI-regenerated.
        """
        ADJ = set(PALETTE) | {"big", "small", "single", "colored", "cartoon",
                              "standing", "fresh", "whole", "sitting"}
        STOP = {"image", "picture", "photo", "scene", "containing", "composition",
                "collage", "side", "version", "copy", "new", "one", "two", "three"}
        out = []
        for m in re.finditer(
                r"\b(?:one|two|three|\d+|a|an)\b((?:\s+[A-Za-z]+){1,2})", prompt):
            words = m.group(1).split()
            phrase = " ".join(words)
            head = next((w.lower() for w in words if w.lower() not in ADJ), None)
            if not head or head in STOP:
                continue
            stem = lambda s: s[:-1] if s.endswith("s") and not s.endswith("ss") else s
            if not any(stem(head) == stem(n) or stem(head) in stem(n)
                       or stem(n) in stem(head) for n in self.items):
                out.append(phrase)
        return out

    def save(self, name, path):
        self.items[name].image.save(path)


def _figure(draw, box_color, dress=None):
    x0, y0 = 150, 40  # cartoon person inside a 400x400 frame
    draw.ellipse([x0 + 20, y0 + 20, x0 + 80, y0 + 80], fill=PALETTE["skin"])     # head
    draw.arc([x0 + 20, y0 + 20, x0 + 80, y0 + 55], 180, 360, fill="black", width=8)  # hair
    draw.rectangle([x0 + 15, y0 + 70, x0 + 85, y0 + 185], fill=box_color)          # torso
    if dress:  # girl: trapezoid dress hides the torso; arms overlap its edge
        draw.polygon([(x0 + 15, y0 + 78), (x0 + 85, y0 + 78),
                      (x0 + 110, y0 + 196), (x0 - 10, y0 + 196)], fill=dress)
        draw.rectangle([x0 + 4, y0 + 95, x0 + 18, y0 + 160], fill=PALETTE["skin"])
        draw.rectangle([x0 + 82, y0 + 95, x0 + 96, y0 + 160], fill=PALETTE["skin"])
        draw.rectangle([x0 + 25, y0 + 195, x0 + 45, y0 + 265], fill=PALETTE["skin"])
        draw.rectangle([x0 + 55, y0 + 195, x0 + 75, y0 + 265], fill=PALETTE["skin"])
        return
    draw.rectangle([x0 + 10, y0 + 95, x0 + 16, y0 + 160], fill=PALETTE["skin"])    # arms
    draw.rectangle([x0 + 84, y0 + 95, x0 + 90, y0 + 160], fill=PALETTE["skin"])
    # boy: pants
    draw.rectangle([x0 + 20, y0 + 185, x0 + 48, y0 + 265], fill=PALETTE["navy"])
    draw.rectangle([x0 + 52, y0 + 185, x0 + 80, y0 + 265], fill=PALETTE["navy"])


def make_test_images():
    """Deterministic 400x400 catalog images for every scenario in the brief."""
    W = lambda: Image.new("RGB", (400, 400), "white")
    imgs = {}
    im = W()
    d = ImageDraw.Draw(im)
    d.ellipse([140, 150, 260, 290], fill=PALETTE["red"])                     # apple
    d.rectangle([196, 118, 204, 152], fill=PALETTE["brown"])
    d.ellipse([206, 122, 246, 148], fill=PALETTE["green"])
    imgs["apple"] = im
    im = W()
    d = ImageDraw.Draw(im)
    d.ellipse([140, 160, 260, 280], fill=PALETTE["orange"])                   # orange
    d.rectangle([196, 132, 204, 162], fill=PALETTE["brown"])
    imgs["orange"] = im
    im = W()
    _figure(ImageDraw.Draw(im), PALETTE["blue"])
    imgs["boy"] = im                                                          # cartoon boy
    im = W()
    _figure(ImageDraw.Draw(im), PALETTE["skin"], dress=PALETTE["pink"])
    imgs["girl"] = im                                                         # cartoon girl
    im = W()
    d = ImageDraw.Draw(im)                                                    # shapes
    d.ellipse([30, 30, 130, 130], fill=PALETTE["red"])
    d.rectangle([160, 30, 260, 130], fill=PALETTE["blue"])
    d.polygon([(340, 35), (295, 125), (385, 125)], fill=PALETTE["green"])
    d.rectangle([30, 180, 200, 260], fill=PALETTE["purple"])
    imgs["shapes"] = im
    im = W()
    d = ImageDraw.Draw(im)                                                    # colors
    for i, c in enumerate(["red", "blue", "green", "yellow", "purple"]):
        d.rectangle([15 + i * 76, 100, 80 + i * 76, 250], fill=PALETTE[c])
    imgs["colors"] = im
    im = W()                                                                  # apple_orange
    im.paste(imgs["apple"].resize((180, 180)), (20, 110))
    im.paste(imgs["orange"].resize((180, 180)), (200, 110))
    imgs["apple_orange"] = im
    im = W()
    d = ImageDraw.Draw(im)                                                    # overlapping
    d.ellipse([80, 80, 240, 240], fill=PALETTE["red"])
    d.rectangle([160, 140, 320, 300], fill=PALETTE["blue"])
    d.polygon([(200, 60), (120, 220), (290, 220)], fill=PALETTE["green"])
    imgs["overlapping_shapes"] = im
    imgs["blank"] = W()                                                       # no objects
    return imgs


def annotate(entry):
    """Copy of the image with every detected object's bbox + label drawn."""
    im = entry.image.copy()
    d = ImageDraw.Draw(im)
    for o in entry.objects:
        x, y, w, h = o.bbox
        d.rectangle([x, y, x + w, y + h], outline="red", width=3)
        d.text((x + 2, max(0, y - 12)), f"{o.name} {o.confidence}", fill="red")
    return im


def demo():
    """One runnable check covering all 7 test scenarios in the brief."""
    cat, imgs = Catalog(), make_test_images()
    for n, im in imgs.items():
        if n != "blank":
            cat.add_image(n, im)
    # 1. successful detection: names, colors, confidences, bboxes
    assert len(cat.items["apple"].objects) == 1
    assert cat.items["apple"].objects[0].color == "red"
    assert len(cat.items["shapes"].objects) == 4, len(cat.items["shapes"].objects)
    assert len(cat.items["colors"].objects) == 5
    assert len(cat.items["apple_orange"].objects) == 2
    assert len(cat.items["boy"].objects) == 1 and "blue" in cat.items["boy"].objects[0].colors
    assert len(cat.items["girl"].objects) == 1
    ov = cat.items["overlapping_shapes"].objects
    assert len(ov) >= 2 and any(_iou(a.bbox, b.bbox) > 0 for a in ov for b in ov if a is not b)
    print("1. detection ok:", {n: [(o.color, o.shape, o.bbox) for o in e.objects]
                               for n, e in cat.items.items()})
    # 2/3. failed detection (blank) -> manual add; then edit
    cat.add_image("blank", imgs["blank"])
    assert cat.items["blank"].objects == [] and cat.items["blank"].status == "no_objects"
    o = cat.add_object("blank", "apple", (140, 150, 121, 141), "red")
    cat.edit_object("blank", o.id, name="red apple", confidence=0.99,
                    bbox=(140, 150, 121, 141))
    assert cat.items["blank"].objects[0].name == "red apple"
    try:
        cat.edit_object("blank", o.id, shape="circle")  # auto-only -> refused
        raise AssertionError("auto field edit should fail")
    except ValueError:
        pass
    print("2/3. failed detection + manual add/edit ok")
    # 4. corrections persist until explicit rerun
    before = (cat.items["blank"].objects[0].name, cat.items["blank"].objects[0].bbox)
    cat.add_image("apple2", imgs["apple"])  # unrelated ingest changes nothing
    assert (cat.items["blank"].objects[0].name, cat.items["blank"].objects[0].bbox) == before
    assert cat.items["blank"].objects[0].corrected
    cat.rerun_detection("blank")  # explicit rerun discards the manual fix
    assert cat.items["blank"].objects == []
    del cat.items["apple2"]
    print("4. manual corrections persist until rerun ok")
    # 5. composite from existing assets, using the real pixels
    comp = cat.generate_composite(
        "Create an image containing one red apple and one orange", "comp1")
    assert comp.source["sources"] == ["apple", "orange"]
    px = np.asarray(comp.image.convert("RGB")).reshape(-1, 3)
    assert ((abs(px - PALETTE["red"]) < 30).all(axis=1)).any()
    assert ((abs(px - PALETTE["orange"]) < 30).all(axis=1)).any()
    try:  # missing assets are reported, never silently AI-generated
        cat.generate_composite("one apple and one dragon fruit", "bad")
        raise AssertionError("missing asset should fail")
    except ValueError as e:
        assert "dragon" in str(e)
    print("5. apple+orange composite ok:", comp.source["sources"])
    # 6. boy + girl + shapes composite
    comp2 = cat.generate_composite(
        "a scene with a boy, a girl and colored shapes", "comp2")
    assert comp2.source["sources"] == ["boy", "girl", "shapes"], comp2.source["sources"]
    assert len(comp2.objects) >= 3  # detection also runs on composites
    print("6. boy+girl+shapes composite ok:", comp2.source["sources"])
    # 7. overlapping + empty covered above (assertions on overlapping_shapes/blank)
    here = os.path.dirname(os.path.abspath(__file__))
    for n, e in cat.items.items():
        annotate(e).save(os.path.join(here, f"det_{n}.png"))  # visual proof
    for n in ("comp1", "comp2"):
        cat.save(n, os.path.join(here, f"{n}.png"))
    print("7. overlapping + empty ok; saved det_*.png + comp1.png comp2.png")


if __name__ == "__main__":
    demo()
