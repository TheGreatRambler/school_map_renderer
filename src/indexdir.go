package pbf

// indexdir.go — an index directory for one PBF extract.
//
//	<baseDir>/<md5 of pbf>/
//	    meta.json               grid, bounds, counts, per-tile file list
//	    lookup.bin              fixed 32-byte records, random access by ordinal
//	    tile_<c>_<r>.rtree      one serialized R-tree per gigantic lat/lng tile
//	    oversized/<ordinal>     48-byte record for entries too big for tiles
//
// Entries get a dense, incrementing 32-bit ordinal in file order. The tile
// R-trees store only (bbox, ordinal); lookup.bin maps ordinal -> (kind, OSM id,
// bbox, location in the PBF) with fixed-size records addressable as
// ordinal * lookupEntrySize, so lookups are a single ReadAt. An entry whose
// box overlaps more than MaxOverlap tiles is written to oversized/ instead of
// being copied into tiles.
//
// Tiles are plain lat/lng rectangles (TileW x TileH degrees) anchored at the
// extract's data bounds. For japan-261116.osm.pbf (~375M entries, ~15 GB of
// raw tree data) the default 8x8 degree tiles give 20 tiles averaging ~750 MB.

import (
	"bufio"
	"crypto/md5"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"

	"github.com/tidwall/rtree"
)

const (
	// lookup record: kind(1) flags(1) pad(2) osmid(8) bbox f32x4 (16)
	//                blob off (8) blob idx (4) pad (8)
	lookupEntrySize = 48
	lookupFlagOver  = 1

	// oversized record: kind(1) flags(1) pad(6) osmid(8) bbox f64x4 (32)
	oversizedRecSize = 48

	// tile leaf spool record: bbox f32x4 relative to the tile center (16)
	// + ordinal uint32 (4). Same numbers the serialized tree stores.
	spoolRecSize = 20
)

// IndexOpts configures BuildIndex.
type IndexOpts struct {
	TileW, TileH float64 // gigantic tile size in degrees (default 8 x 8)
	MaxOverlap   int     // entries overlapping more tiles go to oversized/ (default 2)
	Jobs         int     // parallel tile builders (default 2)
	Force        bool    // rebuild even if the directory exists
}

func (o *IndexOpts) fill() {
	if o.TileW <= 0 {
		o.TileW = 8
	}
	if o.TileH <= 0 {
		o.TileH = 8
	}
	if o.MaxOverlap < 1 {
		o.MaxOverlap = 2
	}
	if o.Jobs < 1 {
		if n := runtime.NumCPU(); n > 4 {
			o.Jobs = 4
		} else {
			o.Jobs = n
		}
	}
}

// TileMeta describes one non-empty tile in the index directory.
type TileMeta struct {
	C       int        `json:"c"`
	R       int        `json:"r"`
	Box     [4]float64 `json:"box"` // minLon, minLat, maxLon, maxLat
	Entries int64      `json:"entries"`
	File    string     `json:"file"`
}

// IndexMeta is the content of meta.json.
type IndexMeta struct {
	MD5        string     `json:"md5"`
	PBF        string     `json:"pbf"`
	Bounds     [4]float64 `json:"bounds"`
	TileW      float64    `json:"tile_w"`
	TileH      float64    `json:"tile_h"`
	Cols       int        `json:"cols"`
	Rows       int        `json:"rows"`
	MaxOverlap int        `json:"max_overlap"`
	Total      int64      `json:"total"` // entries indexed (ordinals assigned)
	Nodes      int64      `json:"nodes"`
	Ways       int64      `json:"ways"`
	Rels       int64      `json:"rels"`
	Duplicated int64      `json:"duplicated"` // extra tile copies
	Oversized  [4]int64   `json:"oversized"`  // per Kind
	LookupSize int        `json:"lookup_entry_size"`
	Tiles      []TileMeta `json:"tiles"`
}

// Index is an open index directory.
type Index struct {
	dir    string
	meta   IndexMeta
	tiles  map[int]*RTreeFile
	over   []oversizedEntry
	paths  map[int]string
	lookup *os.File // persistent handle: ByOrdinal is on the hot path
}

type oversizedEntry struct {
	ord   uint32
	kind  Kind
	osmid int64
	box   [4]float64
}

// IndexDir returns the directory for a PBF: baseDir/<md5>.
func IndexDir(baseDir, pbfPath string) (dir, md5hex string, err error) {
	md5hex, err = fileMD5(pbfPath)
	if err != nil {
		return "", "", err
	}
	return filepath.Join(baseDir, md5hex), md5hex, nil
}

func fileMD5(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := md5.New()
	if _, err := io.CopyBuffer(h, f, make([]byte, 1<<20)); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// BuildIndex builds (or reuses) the index directory for pbfPath and returns
// the directory and its metadata. Progress goes to stderr.
func BuildIndex(baseDir, pbfPath string, opts IndexOpts) (string, *IndexMeta, error) {
	opts.fill()
	dir, md5hex, err := IndexDir(baseDir, pbfPath)
	if err != nil {
		return "", nil, err
	}
	if !opts.Force {
		if m, err := os.ReadFile(filepath.Join(dir, "meta.json")); err == nil {
			var meta IndexMeta
			if json.Unmarshal(m, &meta) == nil {
				return dir, &meta, nil
			}
		}
	}

	if err := os.RemoveAll(dir); err != nil {
		return "", nil, err
	}
	work := dir + ".building"
	if err := os.RemoveAll(work); err != nil {
		return "", nil, err
	}
	if err := os.MkdirAll(filepath.Join(work, "oversized"), 0o755); err != nil {
		return "", nil, err
	}

	bounds, err := indexBounds(pbfPath)
	if err != nil {
		return "", nil, err
	}
	cols := maxInt(1, int(math.Ceil((bounds[2]-bounds[0])/opts.TileW)))
	rows := maxInt(1, int(math.Ceil((bounds[3]-bounds[1])/opts.TileH)))
	fmt.Fprintf(os.Stderr, "md5 %s\ntiles: %dx%d of %.2fx%.2f deg over %.4f,%.4f .. %.4f,%.4f\n",
		md5hex, cols, rows, opts.TileW, opts.TileH, bounds[0], bounds[1], bounds[2], bounds[3])

	b := &builder{
		opts:   opts,
		work:   work,
		bounds: bounds,
		cols:   cols,
		rows:   rows,
	}
	if err := b.run(pbfPath); err != nil {
		os.RemoveAll(work)
		return "", nil, err
	}

	meta := &b.meta
	meta.MD5 = md5hex
	meta.PBF = pbfPath
	meta.Bounds = bounds
	meta.TileW, meta.TileH = opts.TileW, opts.TileH
	meta.Cols, meta.Rows = cols, rows
	meta.MaxOverlap = opts.MaxOverlap
	meta.LookupSize = lookupEntrySize
	mj, _ := json.MarshalIndent(meta, "", "  ")
	if err := os.WriteFile(filepath.Join(work, "meta.json"), mj, 0o644); err != nil {
		return "", nil, err
	}
	if err := os.Rename(work, dir); err != nil {
		return "", nil, err
	}
	return dir, meta, nil
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// indexBounds returns the extract's data bounds, padded slightly.
func indexBounds(pbfPath string) ([4]float64, error) {
	f, err := os.Open(pbfPath)
	if err != nil {
		return [4]float64{}, err
	}
	defer f.Close()
	r, err := NewReader(f)
	if err != nil {
		return [4]float64{}, err
	}
	defer r.Close()
	if h := r.Header(); h.BBox != nil {
		pad := 0.01
		return [4]float64{
			h.BBox.Left - pad, h.BBox.Bottom - pad,
			h.BBox.Right + pad, h.BBox.Top + pad,
		}, nil
	}
	// No header bbox: scan the node section.
	b := [4]float64{180, 90, -180, -90}
	n := 0
	for {
		e, err := r.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return b, err
		}
		if e.Type != Node {
			break
		}
		n++
		b[0] = math.Min(b[0], e.Lon)
		b[1] = math.Min(b[1], e.Lat)
		b[2] = math.Max(b[2], e.Lon)
		b[3] = math.Max(b[3], e.Lat)
	}
	if n == 0 {
		return b, fmt.Errorf("%w: no nodes in %s", ErrBadIndex, pbfPath)
	}
	return [4]float64{b[0] - 0.01, b[1] - 0.01, b[2] + 0.01, b[3] + 0.01}, nil
}

type builder struct {
	opts        IndexOpts
	work        string
	bounds      [4]float64
	cols, rows  int
	meta        IndexMeta
	tileEntries map[int]int64
	tileSpools  map[int]*spoolWriter
	tileCenters map[int][2]float64
}

// tileRect returns the absolute rectangle of tile (c, r).
func (b *builder) tileRect(c, r int) [4]float64 {
	return [4]float64{
		b.bounds[0] + float64(c)*b.opts.TileW,
		b.bounds[1] + float64(r)*b.opts.TileH,
		math.Min(b.bounds[0]+float64(c+1)*b.opts.TileW, b.bounds[2]),
		math.Min(b.bounds[1]+float64(r+1)*b.opts.TileH, b.bounds[3]),
	}
}

type spoolWriter struct {
	f  *os.File
	bw *bufio.Writer
}

func (b *builder) tileIndex(lon, lat float64) (c, r int) {
	c = int(math.Floor((lon - b.bounds[0]) / b.opts.TileW))
	r = int(math.Floor((lat - b.bounds[1]) / b.opts.TileH))
	c = clampInt(c, 0, b.cols-1)
	r = clampInt(r, 0, b.rows-1)
	return
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func (b *builder) run(pbfPath string) error { //nolint:funlen // one streaming pass
	lookup, err := os.Create(filepath.Join(b.work, "lookup.bin"))
	if err != nil {
		return err
	}
	lw := bufio.NewWriterSize(lookup, 1<<20)
	defer func() {
		lookup.Close()
	}()

	b.tileEntries = map[int]int64{}
	b.tileSpools = map[int]*spoolWriter{}
	b.tileCenters = map[int][2]float64{}
	defer func() {
		for _, s := range b.tileSpools {
			s.f.Close()
		}
	}()

	var ord uint32
	err = ForEachEntryBox(pbfPath, func(k Kind, id int64, box [4]float64, loc BlobRef) error {
		if int64(ord) >= 1<<32 {
			return fmt.Errorf("pbf: more than 2^32 entries")
		}
		ordNum := ord
		ord++

		switch k {
		case Node:
			b.meta.Nodes++
		case Way:
			b.meta.Ways++
		case Relation:
			b.meta.Rels++
		}

		// Tile overlap.
		c0, r0 := b.tileIndex(box[0], box[1])
		c1, r1 := b.tileIndex(box[2], box[3])
		span := (c1 - c0 + 1) * (r1 - r0 + 1)

		flags := byte(0)
		if span > b.opts.MaxOverlap {
			flags = lookupFlagOver
			b.meta.Oversized[k]++
		}

		// lookup.bin record
		var rec [lookupEntrySize]byte
		rec[0] = byte(k)
		rec[1] = flags
		binary.LittleEndian.PutUint64(rec[4:], uint64(id))
		for i := 0; i < 4; i++ {
			binary.LittleEndian.PutUint32(rec[12+i*4:], math.Float32bits(float32(box[i])))
		}
		binary.LittleEndian.PutUint64(rec[28:], uint64(loc.Off))
		binary.LittleEndian.PutUint32(rec[36:], uint32(loc.Idx))
		if _, err := lw.Write(rec[:]); err != nil {
			return err
		}

		if flags != 0 {
			// oversized/<ordinal>
			var orec [oversizedRecSize]byte
			orec[0] = byte(k)
			orec[1] = lookupFlagOver
			binary.LittleEndian.PutUint64(orec[8:], uint64(id))
			for i := 0; i < 4; i++ {
				binary.LittleEndian.PutUint64(orec[16+i*8:], math.Float64bits(box[i]))
			}
			path := filepath.Join(b.work, "oversized", fmt.Sprintf("%d", ordNum))
			if err := os.WriteFile(path, orec[:], 0o644); err != nil {
				return err
			}
			return nil
		}

		// Tile spools: box relative to the tile center, float32.
		b.meta.Duplicated += int64(span) - 1
		for r := r0; r <= r1; r++ {
			for c := c0; c <= c1; c++ {
				ti := r*b.cols + c
				s, err := b.spool(ti)
				if err != nil {
					return err
				}
				tc := b.tileCenters[ti]
				var srec [spoolRecSize]byte
				for i := 0; i < 4; i++ {
					v := box[i]
					if i%2 == 0 {
						v -= tc[0]
					} else {
						v -= tc[1]
					}
					binary.LittleEndian.PutUint32(srec[i*4:], math.Float32bits(float32(v)))
				}
				binary.LittleEndian.PutUint32(srec[16:], ordNum)
				if _, err := s.bw.Write(srec[:]); err != nil {
					return err
				}
				b.tileEntries[ti]++
			}
		}

		b.meta.Total++
		if b.meta.Total%5_000_000 == 0 {
			fmt.Fprintf(os.Stderr, "\rindexed %d entries (%d oversized)", b.meta.Total, b.meta.Oversized[Node]+b.meta.Oversized[Way]+b.meta.Oversized[Relation])
		}
		return nil
	})
	if err != nil {
		return err
	}
	if err := lw.Flush(); err != nil {
		return err
	}
	if err := lookup.Close(); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "\nindexed %d entries: %d nodes, %d ways, %d rels; %d oversized, %d extra tile copies\n",
		b.meta.Total, b.meta.Nodes, b.meta.Ways, b.meta.Rels,
		b.meta.Oversized[1]+b.meta.Oversized[2]+b.meta.Oversized[3], b.meta.Duplicated)

	// Flush the tile spools before the tile builders read them.
	for _, s := range b.tileSpools {
		if err := s.bw.Flush(); err != nil {
			return err
		}
	}
	return b.buildTiles()
}

func (b *builder) spool(ti int) (*spoolWriter, error) {
	if s := b.tileSpools[ti]; s != nil {
		return s, nil
	}
	if _, ok := b.tileCenters[ti]; !ok {
		rect := b.tileRect(ti%b.cols, ti/b.cols)
		b.tileCenters[ti] = [2]float64{(rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2}
	}
	f, err := os.Create(filepath.Join(b.work, fmt.Sprintf("spool_%d.bin", ti)))
	if err != nil {
		return nil, err
	}
	s := &spoolWriter{f: f, bw: bufio.NewWriterSize(f, 1<<20)}
	b.tileSpools[ti] = s
	return s, nil
}

// buildTiles turns every spool into a serialized R-tree file, Jobs at a time.
func (b *builder) buildTiles() error {
	type job struct {
		c, r, ti int
		entries  int64
	}
	var jobs []job
	for ti := 0; ti < b.cols*b.rows; ti++ {
		if n := b.tileEntries[ti]; n > 0 {
			jobs = append(jobs, job{ti % b.cols, ti / b.cols, ti, n})
		}
	}

	errs := make(chan error, len(jobs))
	sem := make(chan struct{}, b.opts.Jobs)
	var wg sync.WaitGroup
	for _, j := range jobs {
		wg.Add(1)
		go func(j job) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			errs <- b.buildOne(j.c, j.r, j.ti, j.entries)
		}(j)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			return err
		}
	}
	for _, s := range b.tileSpools {
		s.bw.Flush()
		s.f.Close()
		os.Remove(s.f.Name())
	}
	return nil
}

func (b *builder) buildOne(c, r, ti int, entries int64) error {
	name := fmt.Sprintf("tile_%02d_%02d.rtree", c, r)
	tr := new(rtree.RTree)
	f, err := os.Open(filepath.Join(b.work, fmt.Sprintf("spool_%d.bin", ti)))
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	defer f.Close()
	br := bufio.NewReaderSize(f, 1<<20)

	// Spool records are float32-relative to the tile center; convert back to
	// absolute for the in-memory tree.
	rect := b.tileRect(c, r)
	cx, cy := (rect[0]+rect[2])/2, (rect[1]+rect[3])/2
	var rec [spoolRecSize]byte
	for {
		if _, err := io.ReadFull(br, rec[:]); err != nil {
			if err == io.EOF {
				break
			}
			return err
		}
		minLon := float64(math.Float32frombits(binary.LittleEndian.Uint32(rec[0:]))) + cx
		minLat := float64(math.Float32frombits(binary.LittleEndian.Uint32(rec[4:]))) + cy
		maxLon := float64(math.Float32frombits(binary.LittleEndian.Uint32(rec[8:]))) + cx
		maxLat := float64(math.Float32frombits(binary.LittleEndian.Uint32(rec[12:]))) + cy
		ord := binary.LittleEndian.Uint32(rec[16:])
		tr.Insert([2]float64{minLon, minLat}, [2]float64{maxLon, maxLat}, uint64(ord))
	}

	box := rect
	if err := serializeRTree(tr, filepath.Join(b.work, name), uint64(entries), box, cx, cy); err != nil {
		return err
	}
	st, _ := os.Stat(filepath.Join(b.work, name))
	fmt.Fprintf(os.Stderr, "tile %d,%d: %d entries, %.1f MB (%s)\n",
		c, r, entries, float64(st.Size())/1e6, name)
	b.meta.Tiles = append(b.meta.Tiles, TileMeta{C: c, R: r, Box: box, Entries: entries, File: name})
	return nil
}

// OpenIndex opens an existing index directory.
func OpenIndex(dir string) (*Index, error) {
	mj, err := os.ReadFile(filepath.Join(dir, "meta.json"))
	if err != nil {
		return nil, err
	}
	var meta IndexMeta
	if err := json.Unmarshal(mj, &meta); err != nil {
		return nil, fmt.Errorf("%w: meta.json: %v", ErrBadIndex, err)
	}
	lf, err := os.Open(filepath.Join(dir, "lookup.bin"))
	if err != nil {
		return nil, err
	}
	ix := &Index{dir: dir, meta: meta, tiles: map[int]*RTreeFile{}, paths: map[int]string{}, lookup: lf}
	for _, t := range meta.Tiles {
		ix.paths[t.R*meta.Cols+t.C] = t.File
	}

	// Cache the oversized set.
	entries, err := os.ReadDir(filepath.Join(dir, "oversized"))
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}
	for _, ent := range entries {
		var ord uint32
		if _, err := fmt.Sscanf(ent.Name(), "%d", &ord); err != nil {
			return nil, fmt.Errorf("%w: bad oversized name %q", ErrBadIndex, ent.Name())
		}
		data, err := os.ReadFile(filepath.Join(dir, "oversized", ent.Name()))
		if err != nil {
			return nil, err
		}
		if len(data) != oversizedRecSize {
			return nil, fmt.Errorf("%w: short oversized record %q", ErrBadIndex, ent.Name())
		}
		e := oversizedEntry{
			ord:   ord,
			kind:  Kind(data[0]),
			osmid: int64(binary.LittleEndian.Uint64(data[8:])),
		}
		for i := 0; i < 4; i++ {
			e.box[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[16+i*8:]))
		}
		ix.over = append(ix.over, e)
	}
	return ix, nil
}

// Dir returns the index directory.
func (ix *Index) Dir() string { return ix.dir }

// Meta returns the index metadata.
func (ix *Index) Meta() *IndexMeta { return &ix.meta }

// Close releases open tile files.
func (ix *Index) Close() error {
	var err error
	if ix.lookup != nil {
		if e := ix.lookup.Close(); e != nil && err == nil {
			err = e
		}
	}
	for _, t := range ix.tiles {
		if e := t.Close(); e != nil && err == nil {
			err = e
		}
	}
	return err
}

func (ix *Index) tile(c, r int) (*RTreeFile, error) {
	ti := r*ix.meta.Cols + c
	if t := ix.tiles[ti]; t != nil {
		return t, nil
	}
	name, ok := ix.paths[ti]
	if !ok {
		return nil, nil // empty tile
	}
	t, err := OpenRTreeFile(filepath.Join(ix.dir, name))
	if err != nil {
		return nil, err
	}
	ix.tiles[ti] = t
	return t, nil
}

// ByOrdinal reads the lookup record for an ordinal (a single pread on the
// persistent lookup handle; safe for concurrent use).
func (ix *Index) ByOrdinal(ord uint32) (kind Kind, osmid int64, box [4]float64, loc BlobRef, oversized bool, err error) {
	var rec [lookupEntrySize]byte
	if ix.lookup == nil {
		f, oerr := os.Open(filepath.Join(ix.dir, "lookup.bin"))
		if oerr != nil {
			err = oerr
			return
		}
		ix.lookup = f
	}
	if _, err = ix.lookup.ReadAt(rec[:], int64(ord)*lookupEntrySize); err != nil {
		return
	}
	kind = Kind(rec[0])
	oversized = rec[1]&lookupFlagOver != 0
	osmid = int64(binary.LittleEndian.Uint64(rec[4:]))
	for i := 0; i < 4; i++ {
		box[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(rec[12+i*4:])))
	}
	loc.Off = int64(binary.LittleEndian.Uint64(rec[28:]))
	loc.Idx = int32(binary.LittleEndian.Uint32(rec[36:]))
	return
}

// ResolveNodeIDs returns coordinates for node IDs by looking them up in the
// sorted node section of lookup.bin (ordinals [0, Nodes), ascending by OSM
// node ID — PBF extracts are sorted type-then-ID). IDs that are missing or
// not nodes are simply absent from the result. The sweep reads lookup.bin
// sequentially in chunks, so batches of clustered IDs cost little more than
// the touched byte ranges.
func (ix *Index) ResolveNodeIDs(ids []int64) (map[int64][2]float64, error) {
	if len(ids) == 0 || ix.meta.Nodes == 0 {
		return nil, nil
	}
	if ix.lookup == nil {
		f, err := os.Open(filepath.Join(ix.dir, "lookup.bin"))
		if err != nil {
			return nil, err
		}
		ix.lookup = f
	}

	want := append([]int64(nil), ids...)
	sort.Slice(want, func(i, j int) bool { return want[i] < want[j] })
	uniq := want[:0]
	for i, id := range want {
		if i == 0 || id != want[i-1] {
			uniq = append(uniq, id)
		}
	}

	const chunkRecords = 4096 // 192 KB of 48-byte records
	buf := make([]byte, chunkRecords*lookupEntrySize)
	bufStart := int64(-1) // ordinal of buf[0]
	record := func(ord int64) ([]byte, error) {
		if bufStart >= 0 && ord >= bufStart && ord < bufStart+chunkRecords {
			return buf[(ord-bufStart)*lookupEntrySize:][:lookupEntrySize], nil
		}
		bufStart = ord
		if _, err := ix.lookup.ReadAt(buf, ord*lookupEntrySize); err != nil {
			return nil, err
		}
		return buf[:lookupEntrySize], nil
	}
	osmidAt := func(b []byte) int64 { return int64(binary.LittleEndian.Uint64(b[4:])) }

	res := make(map[int64][2]float64, len(uniq))
	lo, hi := int64(0), ix.meta.Nodes
	for _, id := range uniq {
		l, r := lo, hi
		for l < r {
			mid := (l + r) / 2
			b, err := record(mid)
			if err != nil {
				return nil, err
			}
			if osmidAt(b) < id {
				l = mid + 1
			} else {
				r = mid
			}
		}
		if l >= hi {
			continue
		}
		b, err := record(l)
		if err != nil {
			return nil, err
		}
		if osmidAt(b) != id {
			continue
		}
		res[id] = [2]float64{
			float64(math.Float32frombits(binary.LittleEndian.Uint32(b[12:]))),
			float64(math.Float32frombits(binary.LittleEndian.Uint32(b[16:]))),
		}
		lo = l + 1
	}
	return res, nil
}

// Search visits every indexed entry whose stored box intersects the query
// box. fn receives (kind, osm id, box, oversized). Entries duplicated across
// tiles are reported once. Oversized entries are always checked, regardless
// of the query box.
func (ix *Index) Search(minLon, minLat, maxLon, maxLat float64,
	fn func(Kind, int64, [4]float64, BlobRef, bool) bool) (SearchStats, error) {
	var st SearchStats

	c0 := clampInt(int(math.Floor((minLon-ix.meta.Bounds[0])/ix.meta.TileW)), 0, ix.meta.Cols-1)
	c1 := clampInt(int(math.Floor((maxLon-ix.meta.Bounds[0])/ix.meta.TileW)), 0, ix.meta.Cols-1)
	r0 := clampInt(int(math.Floor((minLat-ix.meta.Bounds[1])/ix.meta.TileH)), 0, ix.meta.Rows-1)
	r1 := clampInt(int(math.Floor((maxLat-ix.meta.Bounds[1])/ix.meta.TileH)), 0, ix.meta.Rows-1)

	seen := map[uint32]struct{}{}
	stop := false
	for r := r0; r <= r1 && !stop; r++ {
		for c := c0; c <= c1 && !stop; c++ {
			t, err := ix.tile(c, r)
			if err != nil {
				return st, err
			}
			if t == nil {
				continue
			}
			_, err = t.Scan(minLon, minLat, maxLon, maxLat, func(v uint64) bool {
				ord := uint32(v)
				if _, dup := seen[ord]; dup {
					return true
				}
				seen[ord] = struct{}{}
				kind, osmid, box, loc, over, err := ix.ByOrdinal(ord)
				if err != nil {
					stop = true
					return false
				}
				st.Hits++
				if !fn(kind, osmid, box, loc, over) {
					stop = true
					return false
				}
				return true
			})
			if err != nil {
				return st, err
			}
		}
	}
	if stop {
		return st, nil
	}

	// Oversized entries.
	for i := range ix.over {
		e := &ix.over[i]
		if e.box[2] < minLon || e.box[0] > maxLon || e.box[3] < minLat || e.box[1] > maxLat {
			continue
		}
		st.Hits++
		if !fn(e.kind, e.osmid, e.box, BlobRef{}, true) {
			break
		}
	}
	return st, nil
}
