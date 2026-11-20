package pbf

// rtreefile.go — serialized, seekable R-tree files plus the geometry pass
// that computes a bounding box for every entry of a PBF extract.
//
// Node layout (each node is one self-contained record):
//
//	bbox    [4]float32  (16 bytes)
//	kind    uint8       (0 = leaf, 1 = branch)
//	count   uint16
//	pad     [3]byte
//	entries[count]      (24 bytes: bbox 16 bytes + uint64 value)
//
// Boxes are stored as float32 in a coordinate system centered on the tree's
// tile center (stored in the file header), which keeps ~1 m precision over an
// 8-degree tile at a third of the bytes of float64. A branch's uint64 value is
// the absolute file offset of the child node; a leaf's value is caller-defined
// (the index in indexdir.go stores a 32-bit ordinal, zero-extended). A query
// descends from the root and reads only nodes whose boxes intersect it,
// seeking by offset.

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"

	"github.com/tidwall/rtree"
)

const (
	rtreeMagic     = "OSMRTREE"
	rtreeVersion   = 4
	rtreeHdrSize   = 80
	rtreeNodeSize  = 24
	rtreeEntrySize = 24

	// Query boxes are widened by this many degrees before comparing against
	// the float32-relative stored boxes, so float32 rounding can never hide
	// an intersection (at 8-degree tiles the error is <4e-7 deg).
	boxEps = 1e-6

	maxValueID = 1 << 40
)

var (
	ErrBadIndex = errors.New("pbf: bad index file")
	debugScan   = os.Getenv("PBF_DEBUG_SCAN") != ""
)

// SearchStats reports how much of an index file a scan touched.
type SearchStats struct {
	NodesRead int64
	BytesRead int64
	Hits      int64
}

// RTreeFile is a serialized R-tree on disk. Scan reuses an internal buffer,
// so one RTreeFile must not be scanned from multiple goroutines at once.
type RTreeFile struct {
	f       *os.File
	count   uint64
	root    int64
	bounds  [4]float64 // absolute degrees
	cx, cy  float64    // tile center: origin of the stored float32 boxes
	size    int64
	scratch []byte
}

// OpenRTreeFile opens a serialized R-tree file.
func OpenRTreeFile(path string) (*RTreeFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	var h [rtreeHdrSize]byte
	if _, err := io.ReadFull(f, h[:]); err != nil {
		f.Close()
		return nil, fmt.Errorf("%w: %v", ErrBadIndex, err)
	}
	if string(h[0:8]) != rtreeMagic {
		f.Close()
		return nil, fmt.Errorf("%w: bad magic", ErrBadIndex)
	}
	if v := binary.LittleEndian.Uint32(h[8:]); v != rtreeVersion {
		f.Close()
		return nil, fmt.Errorf("%w: version %d", ErrBadIndex, v)
	}
	r := &RTreeFile{f: f}
	r.cx = math.Float64frombits(binary.LittleEndian.Uint64(h[16:]))
	r.cy = math.Float64frombits(binary.LittleEndian.Uint64(h[24:]))
	r.count = binary.LittleEndian.Uint64(h[32:])
	r.root = int64(binary.LittleEndian.Uint64(h[40:]))
	for i := 0; i < 4; i++ {
		r.bounds[i] = math.Float64frombits(binary.LittleEndian.Uint64(h[48+i*8:]))
	}
	if st, err := f.Stat(); err == nil {
		r.size = st.Size()
	}
	return r, nil
}

// Bounds returns the extent of the stored boxes.
func (r *RTreeFile) Bounds() [4]float64 { return r.bounds }

// Len returns the number of stored entries.
func (r *RTreeFile) Len() uint64 { return r.count }

// Size returns the file size in bytes.
func (r *RTreeFile) Size() int64 { return r.size }

// Close closes the file.
func (r *RTreeFile) Close() error { return r.f.Close() }

// Scan visits every stored entry whose box intersects the query box, passing
// the entry's value to fn. Returning false from fn stops the scan.
func (r *RTreeFile) Scan(minLon, minLat, maxLon, maxLat float64,
	fn func(value uint64) bool) (SearchStats, error) {
	var st SearchStats
	if r.root < 0 {
		return st, nil
	}
	q := [4]float64{
		minLon - boxEps - r.cx, minLat - boxEps - r.cy,
		maxLon + boxEps - r.cx, maxLat + boxEps - r.cy,
	}
	_, err := r.scanAt(r.root, q[0], q[1], q[2], q[3], &st, fn)
	return st, err
}

func (r *RTreeFile) scanAt(off int64, qMinLon, qMinLat, qMaxLon, qMaxLat float64,
	st *SearchStats, fn func(uint64) bool) (bool, error) {
	var hdr [rtreeNodeSize]byte
	if err := r.readAt(hdr[:], off, st); err != nil {
		return false, err
	}
	st.NodesRead++
	minLon, minLat, maxLon, maxLat := rawBox(hdr[0:])
	if maxLon < qMinLon || minLon > qMaxLon || maxLat < qMinLat || minLat > qMaxLat {
		return true, nil
	}
	kind := hdr[16]
	count := int(binary.LittleEndian.Uint16(hdr[17:]))
	if count == 0 {
		return true, nil
	}

	n := int64(count) * rtreeEntrySize
	if cap(r.scratch) < int(n) {
		r.scratch = make([]byte, n)
	}
	buf := r.scratch[:n]
	if err := r.readAt(buf, off+rtreeNodeSize, st); err != nil {
		return false, err
	}

	switch kind {
	case 0: // leaf
		for i := 0; i < count; i++ {
			p := i * rtreeEntrySize
			eMinLon, eMinLat, eMaxLon, eMaxLat := rawBox(buf[p:])
			if eMaxLon < qMinLon || eMinLon > qMaxLon || eMaxLat < qMinLat || eMinLat > qMaxLat {
				continue
			}
			st.Hits++
			if !fn(binary.LittleEndian.Uint64(buf[p+16:])) {
				return false, nil
			}
		}
	case 1: // branch
		// Collect matching children before recursing: the recursion reuses
		// r.scratch, which buf points into.
		var local [64]int64
		kids := local[:0]
		if count > len(local) {
			kids = make([]int64, 0, count)
		}
		for i := 0; i < count; i++ {
			p := i * rtreeEntrySize
			eMinLon, eMinLat, eMaxLon, eMaxLat := rawBox(buf[p:])
			if eMaxLon < qMinLon || eMinLon > qMaxLon || eMaxLat < qMinLat || eMinLat > qMaxLat {
				continue
			}
			kids = append(kids, int64(binary.LittleEndian.Uint64(buf[p+16:])))
		}
		if debugScan {
			fmt.Fprintf(os.Stderr, "scanAt off=%d kind=%d count=%d kids=%d q=[%v %v %v %v]\n",
				off, kind, count, len(kids), qMinLon, qMinLat, qMaxLon, qMaxLat)
		}
		for _, child := range kids {
			if ok, err := r.scanAt(child, qMinLon, qMinLat, qMaxLon, qMaxLat, st, fn); err != nil {
				return false, err
			} else if !ok {
				return false, nil
			}
		}
	default:
		return false, fmt.Errorf("%w: node kind %d", ErrBadIndex, kind)
	}
	return true, nil
}

func (r *RTreeFile) readAt(p []byte, off int64, st *SearchStats) error {
	n, err := r.f.ReadAt(p, off)
	st.BytesRead += int64(n)
	if err != nil {
		if err == io.EOF {
			return fmt.Errorf("%w: short read at %d", ErrBadIndex, off)
		}
		return err
	}
	return nil
}

// ------------------------------------------------------------- serialization

// serializeRTree writes the in-memory tree as one serialized R-tree file.
// Boxes are stored as float32 relative to (cx, cy).
func serializeRTree(tr *rtree.RTree, path string, count uint64, bounds [4]float64, cx, cy float64) error {
	ch := tr.Children(nil, nil)
	if len(ch) != 1 {
		return fmt.Errorf("%w: unexpected rtree root", ErrBadIndex)
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := f.Seek(rtreeHdrSize, io.SeekStart); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	w := &nodeWriter{bw: bufio.NewWriterSize(f, 1<<20), off: rtreeHdrSize, cx: cx, cy: cy}
	root, err := w.writeNode(tr, ch[0].Data, ch[0].Min, ch[0].Max)
	if err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := w.bw.Flush(); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	var h [rtreeHdrSize]byte
	copy(h[0:8], rtreeMagic)
	binary.LittleEndian.PutUint32(h[8:], rtreeVersion)
	binary.LittleEndian.PutUint64(h[16:], math.Float64bits(cx))
	binary.LittleEndian.PutUint64(h[24:], math.Float64bits(cy))
	binary.LittleEndian.PutUint64(h[32:], count)
	binary.LittleEndian.PutUint64(h[40:], uint64(root))
	for i := 0; i < 4; i++ {
		binary.LittleEndian.PutUint64(h[48+i*8:], math.Float64bits(bounds[i]))
	}
	if _, err := f.WriteAt(h[:], 0); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

type nodeWriter struct {
	bw     *bufio.Writer
	off    int64
	cx, cy float64
}

// writeNode writes a node's children first (post-order), then the node, and
// returns the node's absolute file offset.
func (w *nodeWriter) writeNode(tr *rtree.RTree, parent any, min, max [2]float64) (int64, error) {
	kids := tr.Children(parent, nil)
	if len(kids) == 0 {
		return 0, fmt.Errorf("%w: empty node", ErrBadIndex)
	}
	leaf := kids[0].Item

	childOff := make([]int64, len(kids))
	if !leaf {
		for i, k := range kids {
			off, err := w.writeNode(tr, k.Data, k.Min, k.Max)
			if err != nil {
				return 0, err
			}
			childOff[i] = off
		}
	}

	buf := make([]byte, rtreeNodeSize+len(kids)*rtreeEntrySize)
	putRelBox(buf[0:], [2]float64{min[0], min[1]}, [2]float64{max[0], max[1]}, w.cx, w.cy)
	if leaf {
		buf[16] = 0
	} else {
		buf[16] = 1
	}
	binary.LittleEndian.PutUint16(buf[17:], uint16(len(kids)))
	for i, k := range kids {
		p := rtreeNodeSize + i*rtreeEntrySize
		putRelBox(buf[p:], k.Min, k.Max, w.cx, w.cy)
		var v uint64
		if leaf {
			v = k.Data.(uint64)
		} else {
			v = uint64(childOff[i])
		}
		binary.LittleEndian.PutUint64(buf[p+16:], v)
	}
	off := w.off
	if _, err := w.bw.Write(buf); err != nil {
		return 0, err
	}
	w.off += int64(len(buf))
	return off, nil
}

// putRelBox stores a box as float32 offsets from the tile center.
func putRelBox(dst []byte, min, max [2]float64, cx, cy float64) {
	binary.LittleEndian.PutUint32(dst[0:], math.Float32bits(float32(min[0]-cx)))
	binary.LittleEndian.PutUint32(dst[4:], math.Float32bits(float32(min[1]-cy)))
	binary.LittleEndian.PutUint32(dst[8:], math.Float32bits(float32(max[0]-cx)))
	binary.LittleEndian.PutUint32(dst[12:], math.Float32bits(float32(max[1]-cy)))
}

// rawBox reads a float32-center-relative box without translating to absolute.
func rawBox(b []byte) (minLon, minLat, maxLon, maxLat float64) {
	return float64(math.Float32frombits(binary.LittleEndian.Uint32(b[0:]))),
		float64(math.Float32frombits(binary.LittleEndian.Uint32(b[4:]))),
		float64(math.Float32frombits(binary.LittleEndian.Uint32(b[8:]))),
		float64(math.Float32frombits(binary.LittleEndian.Uint32(b[12:])))
}

// relBox reads a float32-center-relative box back into absolute degrees.
func relBox(b []byte, cx, cy float64) (minLon, minLat, maxLon, maxLat float64) {
	return float64(math.Float32frombits(binary.LittleEndian.Uint32(b[0:]))) + cx,
		float64(math.Float32frombits(binary.LittleEndian.Uint32(b[4:]))) + cy,
		float64(math.Float32frombits(binary.LittleEndian.Uint32(b[8:]))) + cx,
		float64(math.Float32frombits(binary.LittleEndian.Uint32(b[12:]))) + cy
}

// ---------------------------------------------------------------- geometry

type nodeStore struct {
	ids  []int64
	lons []int32
	lats []int32
}

type geomStore struct {
	nodes  nodeStore
	wayID  []int64
	wayBox []float64 // 4 per way
}

// ForEachEntryBox streams pbfPath and calls fn with the bounding box of every
// entry: a point for nodes, the union of member boxes for ways and relations.
// Relations that only reference other relations are skipped.
//
// It keeps all node coordinates and way boxes in memory, so it is meant for
// building indexes, not for interactive use.
func ForEachEntryBox(pbfPath string, fn func(Kind, int64, [4]float64, BlobRef) error) error {
	f, err := os.Open(pbfPath)
	if err != nil {
		return err
	}
	defer f.Close()
	r, err := NewReader(f)
	if err != nil {
		return err
	}
	defer r.Close()

	g := &geomStore{}
	for {
		e, err := r.Next()
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		switch e.Type {
		case Node:
			g.nodes.ids = append(g.nodes.ids, e.ID)
			g.nodes.lons = append(g.nodes.lons, int32(math.Round(e.Lon*1e7)))
			g.nodes.lats = append(g.nodes.lats, int32(math.Round(e.Lat*1e7)))
			if err := fn(Node, e.ID, [4]float64{e.Lon, e.Lat, e.Lon, e.Lat}, e.Loc); err != nil {
				return err
			}
		case Way:
			box, ok := g.wayBounds(e.Refs)
			if !ok {
				continue
			}
			g.wayID = append(g.wayID, e.ID)
			g.wayBox = append(g.wayBox, box[0], box[1], box[2], box[3])
			if err := fn(Way, e.ID, box, e.Loc); err != nil {
				return err
			}
		case Relation:
			box, ok := g.relBounds(e.Members)
			if !ok {
				continue
			}
			if err := fn(Relation, e.ID, box, e.Loc); err != nil {
				return err
			}
		}
	}
}

// lookupNode uses interpolation search: node IDs are sorted but very
// non-uniformly distributed (old edits, bulk imports), so plain binary search
// is needlessly slow on large extracts.
func (g *geomStore) lookupNode(id int64) (lon, lat float64, ok bool) {
	ids := g.nodes.ids
	lo, hi := 0, len(ids)-1
	for lo <= hi {
		if ids[lo] == id {
			return float64(g.nodes.lons[lo]) / 1e7, float64(g.nodes.lats[lo]) / 1e7, true
		}
		if ids[hi] == id {
			return float64(g.nodes.lons[hi]) / 1e7, float64(g.nodes.lats[hi]) / 1e7, true
		}
		if hi-lo <= 1 {
			return 0, 0, false
		}
		mid := (lo + hi) / 2
		if den := float64(ids[hi] - ids[lo]); den > 0 {
			frac := float64(id-ids[lo]) / den
			mid = lo + int(frac*float64(hi-lo))
			if mid <= lo {
				mid = lo + 1
			}
			if mid >= hi {
				mid = hi - 1
			}
		}
		switch {
		case ids[mid] < id:
			lo = mid + 1
		case ids[mid] > id:
			hi = mid - 1
		default:
			return float64(g.nodes.lons[mid]) / 1e7, float64(g.nodes.lats[mid]) / 1e7, true
		}
	}
	return 0, 0, false
}

func (g *geomStore) lookupWay(id int64) (box [4]float64, ok bool) {
	i := sort.Search(len(g.wayID), func(i int) bool { return g.wayID[i] >= id })
	if i >= len(g.wayID) || g.wayID[i] != id {
		return box, false
	}
	b := g.wayBox[i*4:]
	return [4]float64{b[0], b[1], b[2], b[3]}, true
}

func (g *geomStore) wayBounds(refs []int64) (box [4]float64, ok bool) {
	first := true
	for _, ref := range refs {
		lon, lat, hit := g.lookupNode(ref)
		if !hit {
			continue
		}
		if first {
			box = [4]float64{lon, lat, lon, lat}
			first = false
			continue
		}
		box[0] = math.Min(box[0], lon)
		box[1] = math.Min(box[1], lat)
		box[2] = math.Max(box[2], lon)
		box[3] = math.Max(box[3], lat)
	}
	return box, !first
}

func (g *geomStore) relBounds(members []Member) (box [4]float64, ok bool) {
	for _, m := range members {
		var b [4]float64
		var hit bool
		switch m.Type {
		case Node:
			lon, lat, found := g.lookupNode(m.Ref)
			if found {
				b = [4]float64{lon, lat, lon, lat}
				hit = true
			}
		case Way:
			b, hit = g.lookupWay(m.Ref)
		}
		if !hit {
			continue
		}
		if !ok {
			box, ok = b, true
			continue
		}
		box[0] = math.Min(box[0], b[0])
		box[1] = math.Min(box[1], b[1])
		box[2] = math.Max(box[2], b[2])
		box[3] = math.Max(box[3], b[3])
	}
	return box, ok
}
