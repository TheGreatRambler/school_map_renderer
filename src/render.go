package pbf

// render.go — rasterize a tile into a PNG, in memory.
//
// A "tile" is defined by a TileRef: a size in degrees plus integer indices
// into the index's bounds, the same coordinates the query side uses. The
// renderer
//
//  1. queries the index over the tile (plus a margin) for bboxes and locations,
//  2. fetches the way entries from the PBF via their blob offsets, batched by
//     block so each block is decoded at most once,
//  3. classifies each feature from its tags (roads, rail, water, buildings,
//     POI nodes, ...),
//  4. rasterizes with scanline polygon fill for closed areas, thick-throat
//     Bresenham lines for everything else, and encodes a PNG with the standard
//     library.
//
// Node coordinates come straight from the lookup bboxes (no node fetching).
// Way geometry is the real polyline, with segments broken where a referenced
// node lies outside the query margin. Relations are skipped.

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"sort"
)

// TileRef identifies a tile: size in degrees plus indices into the index
// bounds (origin = min corner), matching the query-side (-tw -th -x -y).
type TileRef struct {
	W, H float64 // tile size in degrees
	X, Y int     // tile indices
}

// RenderStyle configures the drawing.
type RenderStyle struct {
	Width     int     // unused placeholder for future style knobs
	Margin    float64 // context margin around the tile as a fraction (default 0.25)
	NoOutline bool    // skip the red query-tile outline (for mosaics)
	PoiDots   bool    // draw amenity/shop/leisure nodes as dots (off by default:
	// dense city tiles have one POI per pixel)
}

// Way geometry classes, in draw order (later = on top).
const (
	clsArea = iota
	clsBuilding
	clsWater
	clsPath
	clsMinor
	clsRoad
	clsMajor
	clsRail
	numClasses
)

var classColor = [numClasses]color.RGBA{
	clsArea:     {0xd9, 0xe6, 0xd2, 0xff},
	clsBuilding: {0xd0, 0xcf, 0xce, 0xff},
	clsWater:    {0x9e, 0xc5, 0xe8, 0xff},
	clsPath:     {0xd5, 0xca, 0xac, 0xff},
	clsMinor:    {0xb8, 0xb8, 0xb8, 0xff},
	clsRoad:     {0xe0, 0x9f, 0x5a, 0xff},
	clsMajor:    {0xd9, 0x60, 0x4e, 0xff},
	clsRail:     {0x5c, 0x5c, 0x5c, 0xff},
}

// Base widths at px=1024, scaled with image size.
var classWidth = [numClasses]float64{
	clsArea:     0, // filled
	clsBuilding: 0, // filled
	clsWater:    1.4,
	clsPath:     1,
	clsMinor:    1.6,
	clsRoad:     2.6,
	clsMajor:    4,
	clsRail:     2.4,
}

// Rail and (optionally) motorway casing colors.
var (
	railCasing = color.RGBA{0xf8, 0xf8, 0xf6, 0xff}
	otherColor = color.RGBA{0xcc, 0xcc, 0xcc, 0xff}
	touchColor = color.RGBA{0xd0, 0x30, 0x30, 0xff} // query tile outline
	poiAmens   = color.RGBA{0x4f, 0x8f, 0xd0, 0xff} // amenity/shop
	poiLeisure = color.RGBA{0x4f, 0xa0, 0x60, 0xff} // leisure/tourism
)

func classifyWay(tags map[string]string) int {
	if h, ok := tags["highway"]; ok {
		switch h {
		case "motorway", "motorway_link", "trunk", "trunk_link":
			return clsMajor
		case "primary", "primary_link", "secondary", "secondary_link",
			"tertiary", "tertiary_link":
			return clsRoad
		case "footway", "path", "steps", "cycleway", "track", "bridleway", "corridor":
			return clsPath
		default:
			return clsMinor
		}
	}
	if _, ok := tags["railway"]; ok {
		return clsRail
	}
	if _, ok := tags["waterway"]; ok {
		return clsWater
	}
	if _, ok := tags["natural"]; ok && tags["natural"] == "water" {
		return clsWater
	}
	if _, ok := tags["building"]; ok {
		return clsBuilding
	}
	if _, ok := tags["landuse"]; ok {
		return clsArea
	}
	return clsOther
}

const clsOther = numClasses // drawn last, gray

// isPOI reports whether a node should be drawn as a labeled point.
func isPOI(tags map[string]string) bool {
	for _, k := range []string{"amenity", "shop", "leisure", "tourism"} {
		if _, ok := tags[k]; ok {
			return true
		}
	}
	return false
}

type feature struct {
	cls   int
	pts   [][2]float64 // NaN = break
	area  bool         // closed: fill instead of stroke
	tags  map[string]string
	name  string
	osmid int64
}

// RenderTilePNG renders the tile identified by ref into a px x px PNG and
// returns the encoded bytes.
func (ix *Index) RenderTilePNG(ref TileRef, pbfPath string, px int) ([]byte, error) {
	return ix.RenderTileStyled(ref, pbfPath, px, RenderStyle{})
}

// RenderTileStyled is RenderTilePNG with a style.
func (ix *Index) RenderTileStyled(ref TileRef, pbfPath string, px int, style RenderStyle) ([]byte, error) {
	if px <= 0 {
		px = 1024
	}
	scale := float64(px) / 1024.0

	m := ix.meta
	b := m.Bounds
	box := [4]float64{
		b[0] + float64(ref.X)*ref.W,
		b[1] + float64(ref.Y)*ref.H,
		b[0] + float64(ref.X+1)*ref.W,
		b[1] + float64(ref.Y+1)*ref.H,
	}
	// Search exactly the tile: every entry whose box overlaps the tile is
	// included; nothing outside. Way geometry is completed later by resolving
	// every ref via the sorted node section of lookup.bin.

	// Pass 1: index query. Node points come straight from the lookup bboxes;
	// ways and POI nodes are queued for fetching.
	nodeCoord := make(map[int64][2]float64)
	type fetchReq struct {
		loc BlobRef
		id  int64
	}
	var ways, poiNodes []fetchReq
	_, err := ix.Search(box[0], box[1], box[2], box[3],
		func(k Kind, id int64, box [4]float64, loc BlobRef, over bool) bool {
			if (loc.Off == 0 && loc.Idx == 0) && k != Node {
				return true // oversized entries have no loc; skipped
			}
			switch k {
			case Node:
				nodeCoord[id] = [2]float64{box[0], box[1]}
				if style.PoiDots && (loc.Off != 0 || loc.Idx != 0) {
					poiNodes = append(poiNodes, fetchReq{loc, id})
				}
			case Way:
				if loc.Off != 0 || loc.Idx != 0 {
					ways = append(ways, fetchReq{loc, id})
				}
			}
			return true
		})
	if err != nil {
		return nil, err
	}

	// Pass 2: fetch ways and POI nodes, grouped by block (one decode each).
	// Way refs outside the tile are collected and resolved exactly afterwards.
	type wayData struct {
		cls    int
		refs   []int64
		closed bool
		tags   map[string]string
		osmid  int64
	}
	wdata := make([]wayData, 0, len(ways))
	poiClass := map[int64]bool{}
	missing := make(map[int64]struct{})
	reqs := append(append([]fetchReq(nil), ways...), poiNodes...)
	sort.Slice(reqs, func(i, j int) bool { return reqs[i].loc.Off < reqs[j].loc.Off })
	fetcher, err := NewEntryFetcher(pbfPath)
	if err != nil {
		return nil, err
	}
	defer fetcher.Close()
	for i := 0; i < len(reqs); {
		off := reqs[i].loc.Off
		j := i
		for j < len(reqs) && reqs[j].loc.Off == off {
			j++
		}
		view, err := fetcher.Block(off)
		if err != nil {
			return nil, fmt.Errorf("fetch block at %d: %w", off, err)
		}
		for ; i < j; i++ {
			e, err := view.At(reqs[i].loc.Idx)
			if err != nil {
				return nil, fmt.Errorf("fetch %d: %w", reqs[i].id, err)
			}
			if e.ID != reqs[i].id {
				continue // offset/index mismatch; skip rather than misdraw
			}
			if e.Type == Node {
				if isPOI(e.Tags) {
					if _, l := e.Tags["leisure"]; l {
						poiClass[e.ID] = true
					} else if _, t := e.Tags["tourism"]; t {
						poiClass[e.ID] = true
					}
				}
				continue
			}
			cls := classifyWay(e.Tags)
			closed := len(e.Refs) > 2 && e.Refs[0] == e.Refs[len(e.Refs)-1]
			for _, r := range e.Refs {
				if _, ok := nodeCoord[r]; !ok {
					missing[r] = struct{}{}
				}
			}
			// e.Refs/e.Tags share the fetcher's per-block buffers, which the
			// next Block call overwrites — clone what we keep.
			refs := append([]int64(nil), e.Refs...)
			tags := make(map[string]string, len(e.Tags))
			for k, v := range e.Tags {
				tags[k] = v
			}
			wdata = append(wdata, wayData{cls: cls, refs: refs, closed: closed, tags: tags, osmid: e.ID})
		}
	}

	// Pass 3: complete way geometry. Any ref not covered by the tile's node
	// set is looked up in the sorted node section of lookup.bin, so every
	// feature intersecting the tile is drawn in full (clipped to tile pixels
	// by the rasterizer).
	if len(missing) > 0 {
		ids := make([]int64, 0, len(missing))
		for id := range missing {
			ids = append(ids, id)
		}
		resolved, err := ix.ResolveNodeIDs(ids)
		if err != nil {
			return nil, fmt.Errorf("resolve node refs: %w", err)
		}
		for id, c := range resolved {
			nodeCoord[id] = c
		}
	}

	// Build polylines now that nodeCoord is complete.
	feats := make([]feature, 0, len(wdata))
	for i := range wdata {
		w := &wdata[i]
		pts := make([][2]float64, 0, len(w.refs))
		for _, r := range w.refs {
			if c, ok := nodeCoord[r]; ok {
				pts = append(pts, c)
			} else if len(pts) > 0 && !math.IsNaN(pts[len(pts)-1][0]) {
				pts = append(pts, [2]float64{math.NaN(), math.NaN()}) // break
			}
		}
		feats = append(feats, feature{
			cls:   w.cls,
			pts:   pts,
			area:  w.closed && (w.cls == clsBuilding || w.cls == clsArea || w.cls == clsWater),
			tags:  w.tags,
			name:  w.tags["name"],
			osmid: w.osmid,
		})
	}
	// Areas drawn before everything, then by class order; other/POIs last.
	sort.SliceStable(feats, func(a, b int) bool {
		if (feats[a].area) != (feats[b].area) {
			return feats[a].area
		}
		return feats[a].cls < feats[b].cls
	})

	// Rasterize.
	img := image.NewRGBA(image.Rect(0, 0, px, px))
	bg := color.RGBA{0xf8, 0xf8, 0xf6, 0xff}
	fillRect(img, img.Rect, bg)
	// Floor, not truncation: truncation is not translation-consistent for
	// negative coordinates, which shifts segments crossing tile seams by 1px
	// between the two tiles. floor(f+n) == floor(f)+n exactly.
	toPx := func(lon, lat float64) (int, int) {
		x := int(math.Floor((lon - box[0]) / (box[2] - box[0]) * float64(px)))
		y := int(math.Floor((box[3] - lat) / (box[3] - box[1]) * float64(px)))
		return x, y
	}

	drawFeat := func(f *feature, casing bool) {
		if len(f.pts) < 2 {
			return
		}
		w := 1.2 * scale
		if f.cls < numClasses {
			w = classWidth[f.cls] * scale
		}
		if w < 1 {
			w = 1
		}
		line := otherColor
		if f.cls < numClasses {
			line = classColor[f.cls]
		}
		if casing {
			drawPolyline(img, f.pts, toPx, math.Max(w+1.6*scale, w), railCasing)
		}
		drawPolyline(img, f.pts, toPx, w, line)
	}
	for i := range feats {
		f := &feats[i]
		if f.area {
			drawArea(img, f, toPx, scale)
		}
	}
	for _, cls := range []int{clsWater, clsPath, clsMinor, clsRoad, clsMajor, clsRail, clsOther} {
		for i := range feats {
			if feats[i].area || feats[i].cls != cls {
				continue
			}
			if cls == clsMajor || cls == clsRail {
				drawFeat(&feats[i], true) // casing under the line
			} else {
				drawFeat(&feats[i], false)
			}
		}
	}

	for _, req := range poiNodes {
		c, ok := nodeCoord[req.id]
		if !ok {
			continue
		}
		x, y := toPx(c[0], c[1])
		cx := poiAmens
		if poiClass[req.id] {
			cx = poiLeisure
		}
		fillRect(img, image.Rect(x, y, x+1, y+1), cx)
	}

	// Query tile outline.
	if !style.NoOutline {
		x0, y0 := toPx(box[0], box[3])
		x1, y1 := toPx(box[2], box[1])
		drawLine(img, x0, y0, x1, y0, touchColor, int(1+scale))
		drawLine(img, x1, y0, x1, y1, touchColor, int(1+scale))
		drawLine(img, x1, y1, x0, y1, touchColor, int(1+scale))
		drawLine(img, x0, y1, x0, y0, touchColor, int(1+scale))
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// drawArea fills a closed polygon (even-odd rule) with its class color and
// optionally strokes the outline. NaN breaks split the rings.
func drawArea(img *image.RGBA, f *feature, toPx func(float64, float64) (int, int), scale float64) {
	fill := classColor[f.cls]
	switch f.cls {
	case clsWater:
		fill = classColor[clsWater]
	case clsBuilding:
		fill = color.RGBA{0xc8, 0xc8, 0xcc, 0xff}
	case clsArea:
		fill = classColor[clsArea]
	default:
		fill = otherColor
	}
	fillPolygon(img, f.pts, toPx, fill)
}

// fillPolygon fills closed polygons with the even-odd rule. NaN separates
// rings; each ring must be closed or it is closed implicitly.
func fillPolygon(img *image.RGBA, pts [][2]float64,
	toPx func(float64, float64) (int, int), c color.RGBA) {
	// Project all vertices once; split into rings at NaN.
	type ring []struct{ x, y float64 }
	var rings []ring
	cur := ring{}
	for _, p := range pts {
		if math.IsNaN(p[0]) {
			if len(cur) > 2 {
				rings = append(rings, cur)
			}
			cur = ring{}
			continue
		}
		x, y := toPx(p[0], p[1])
		cur = append(cur, struct{ x, y float64 }{float64(x), float64(y)})
	}
	if len(cur) > 2 {
		rings = append(rings, cur)
	}

	// Scanline with even-odd rule over all rings.
	for sy := 0; sy < img.Rect.Dy(); sy++ {
		yc := float64(sy) + 0.5
		var xs []float64
		for _, rg := range rings {
			n := len(rg)
			for i := 0; i < n; i++ {
				a, b := rg[i], rg[(i+1)%n]
				if (a.y <= yc && b.y > yc) || (b.y <= yc && a.y > yc) {
					t := (yc - a.y) / (b.y - a.y)
					xs = append(xs, a.x+t*(b.x-a.x))
				}
			}
		}
		sort.Float64s(xs)
		for i := 0; i+1 < len(xs); i += 2 {
			x0, x1 := int(math.Ceil(xs[i]-0.5)), int(math.Floor(xs[i+1]-0.5))
			for x := x0; x <= x1; x++ {
				setPx(img, x, sy, c)
			}
		}
	}
}

// drawPolyline strokes pts (NaN = break) using the coordinate transform.
func drawPolyline(img *image.RGBA, pts [][2]float64,
	toPx func(float64, float64) (int, int), width float64, c color.RGBA) {
	thick := int(width)
	if thick < 1 {
		thick = 1
	}
	for s := 0; s < len(pts)-1; s++ {
		if math.IsNaN(pts[s][0]) || math.IsNaN(pts[s+1][0]) {
			continue
		}
		x0, y0 := toPx(pts[s][0], pts[s][1])
		x1, y1 := toPx(pts[s+1][0], pts[s+1][1])
		drawLine(img, x0, y0, x1, y1, c, thick)
	}
}

func fillRect(img *image.RGBA, r image.Rectangle, c color.RGBA) {
	r = r.Intersect(img.Rect)
	for y := r.Min.Y; y < r.Max.Y; y++ {
		row := y*img.Stride + r.Min.X*4
		for x := r.Min.X; x < r.Max.X; x++ {
			i := row + (x-r.Min.X)*4
			img.Pix[i] = c.R
			img.Pix[i+1] = c.G
			img.Pix[i+2] = c.B
			img.Pix[i+3] = c.A
		}
	}
}

// drawLine is Bresenham with a crude thickness (extra parallel passes).
func drawLine(img *image.RGBA, x0, y0, x1, y1 int, c color.RGBA, thick int) {
	dx, dy := abs(x1-x0), -abs(y1-y0)
	sx, sy := 1, 1
	if x0 >= x1 {
		sx = -1
	}
	if y0 >= y1 {
		sy = -1
	}
	err := dx + dy
	for {
		for t := 0; t < thick; t++ {
			// offset perpendicular-ish: spread vertically
			setPx(img, x0, y0+t-thick/2, c)
		}
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			err += dx
			y0 += sy
		}
	}
}

func setPx(img *image.RGBA, x, y int, c color.RGBA) {
	if x < 0 || y < 0 || x >= img.Rect.Dx() || y >= img.Rect.Dy() {
		return
	}
	img.SetRGBA(x, y, c)
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
