// Package pbf implements a streaming, bounded-memory reader for OpenStreetMap
// PBF files (.osm.pbf).
//
// The format is a sequence of length-prefixed blobs. This reader holds at most
// one blob in memory at a time: the compressed blob plus its decompressed
// contents, plus the per-block string table and bookkeeping. Peak memory is
// therefore O(MaxBlob), independent of the file size.
//
// The reader works on one PrimitiveBlock at a time. Within a block it indexes
// the individual objects (cheap slice slicing) and decodes them lazily, one
// per call to Next.
//
// Entries returned by Next share storage with the Reader and are only valid
// until the next call to Next. Use Entry.Clone to keep one.
package pbf

import (
	"bufio"
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

const (
	// MaxBlobHeaderSize is the hard limit from the PBF spec: a BlobHeader
	// (including its length prefix) may not exceed 64 KiB.
	MaxBlobHeaderSize = 64 << 10

	// MaxBlobSize is the hard limit from the PBF spec for a Blob, compressed
	// or uncompressed. Writers target 16 MiB blocks.
	MaxBlobSize = 32 << 20
)

var (
	ErrCorrupt      = errors.New("pbf: corrupt data")
	ErrTruncated    = errors.New("pbf: truncated data")
	ErrBlobTooLarge = errors.New("pbf: blob exceeds size limit")
)

// Kind is the type of an OSM primitive.
type Kind uint8

const (
	Node Kind = iota + 1
	Way
	Relation
)

func (k Kind) String() string {
	switch k {
	case Node:
		return "node"
	case Way:
		return "way"
	case Relation:
		return "relation"
	}
	return "unknown"
}

// Member is one member of a relation.
type Member struct {
	Type Kind
	Ref  int64
	Role string
}

// BlobRef locates an object inside the source PBF: the byte offset of the
// containing blob's length prefix, plus the object's index within that block.
// Offsets are relative to the start of the Reader's input stream, so they are
// file offsets when the Reader was created over a file opened at position 0.
type BlobRef struct {
	Off int64
	Idx int32
}

// Entry is a single OSM object. Its Tags map, Refs/Members slices etc. are
// backed by the Reader's reusable buffers and must not be retained across
// calls to Next. Use Clone to keep an Entry.
type Entry struct {
	Type Kind
	ID   int64

	// Loc is where this object lives in the source stream.
	Loc BlobRef

	// Nodes only.
	Lat, Lon float64

	Tags map[string]string

	// Ways only. Refs are absolute node IDs.
	Refs []int64
	// RefLats/RefLons are set only if the file embeds locations on ways
	// ("locations_on_ways" extension). Usually empty.
	RefLats, RefLons []float64

	// Relations only.
	Members []Member

	Version   int32
	Timestamp int64 // Unix seconds; 0 if the file has no metadata.
}

// Clone returns a deep copy of e that is safe to keep.
func (e *Entry) Clone() Entry {
	c := *e
	if e.Tags != nil {
		c.Tags = make(map[string]string, len(e.Tags))
		for k, v := range e.Tags {
			c.Tags[k] = v
		}
	}
	c.Refs = append([]int64(nil), e.Refs...)
	c.RefLats = append([]float64(nil), e.RefLats...)
	c.RefLons = append([]float64(nil), e.RefLons...)
	c.Members = append([]Member(nil), e.Members...)
	return c
}

// BBox is the header bounding box in degrees.
type BBox struct {
	Left, Right, Top, Bottom float64
}

// Header contains the contents of the OSMHeader block.
type Header struct {
	BBox                 *BBox
	RequiredFeatures     []string
	OptionalFeatures     []string
	WritingProgram       string
	Source               string
	ReplicationTimestamp int64
	ReplicationSequence  int64
	ReplicationURL       string
}

type denseGroup struct {
	ids     []int64
	lats    []int64
	lons    []int64
	kv      []uint64
	kvOff   []int // kvOff[i] = start of node i's key/value pairs in kv
	vectors []uint32
	stamps  []int64 // absolute Unix seconds
}

type objRef struct {
	kind  Kind
	dense int // index into Reader.dense, or -1 for a plain message
	idx   int // index inside the dense group
	b     []byte
}

// Reader is a sequential, one-blob-at-a-time PBF decoder.
type Reader struct {
	br      *bufio.Reader
	maxBlob int

	// Byte position tracking, for Entry.Loc.
	consumed   int64
	curBlobOff int64

	hdr Header

	hdrBuf  []byte
	blobBuf []byte
	dataBuf []byte
	zsrc    bytes.Reader
	zr      io.ReadCloser

	// Per-block data. Valid until the next block is read.
	strings []string
	objs    []objRef
	dense   []denseGroup
	groups  [][]byte

	// Block-level coordinate parameters.
	gran     int64
	dateGran int64
	latOff   int64
	lonOff   int64

	next int
	eof  bool

	// Reusable per-entry storage.
	entry   Entry
	tags    map[string]string
	refs    []int64
	refLats []float64
	refLons []float64
	members []Member

	// Reusable scratch for decoding.
	keys []uint64
	vals []uint64
	ids  []int64
	lats []int64
	lons []int64
	kv   []uint64

	// Stats.
	Blocks            int
	MaxCompressedBlob int
	MaxRawBlob        int
}

// Option configures a Reader.
type Option func(*Reader)

// WithMaxBlob sets the maximum accepted size of a single compressed or
// uncompressed blob. Values smaller than the largest blob in the file cause an
// ErrBlobTooLarge. Default is MaxBlobSize (32 MiB).
func WithMaxBlob(n int) Option {
	return func(r *Reader) {
		if n > 0 {
			r.maxBlob = n
		}
	}
}

// NewReader creates a Reader and consumes the leading OSMHeader block.
func NewReader(in io.Reader, opts ...Option) (*Reader, error) {
	r := &Reader{
		br:      bufio.NewReaderSize(in, 64<<10),
		maxBlob: MaxBlobSize,
		tags:    make(map[string]string, 16),
	}
	for _, opt := range opts {
		opt(r)
	}
	if r.maxBlob < 1<<10 {
		return nil, fmt.Errorf("pbf: max blob size %d is too small", r.maxBlob)
	}
	if err := r.readHeader(); err != nil {
		return nil, err
	}
	return r, nil
}

// Header returns the parsed file header.
func (r *Reader) Header() Header { return r.hdr }

// Close releases the zlib reader, if any.
func (r *Reader) Close() error {
	if r.zr != nil {
		return r.zr.Close()
	}
	return nil
}

// Next returns the next OSM object in file order (all nodes, then all ways,
// then all relations). It returns io.EOF at the end of the file.
func (r *Reader) Next() (*Entry, error) {
	for {
		if r.next < len(r.objs) {
			ref := &r.objs[r.next]
			r.next++
			e, err := r.decode(ref)
			if err == nil && e != nil {
				e.Loc = BlobRef{Off: r.curBlobOff, Idx: int32(r.next - 1)}
			}
			return e, err
		}
		if r.eof {
			return nil, io.EOF
		}
		if err := r.readBlock(); err != nil {
			if err == io.EOF {
				r.eof = true
				return nil, io.EOF
			}
			return nil, err
		}
	}
}

func (r *Reader) readHeader() error {
	typ, data, err := r.readBlob()
	if err != nil {
		return err
	}
	if typ != "OSMHeader" {
		return fmt.Errorf("%w: first block is %q, want OSMHeader", ErrCorrupt, typ)
	}
	return r.parseHeaderBlock(data)
}

func (r *Reader) readBlock() error {
	typ, data, err := r.readBlob()
	if err != nil {
		return err
	}
	if typ == "OSMHeader" {
		return fmt.Errorf("%w: unexpected OSMHeader block", ErrCorrupt)
	}
	if typ != "OSMData" {
		return fmt.Errorf("%w: unknown block type %q", ErrCorrupt, typ)
	}
	r.Blocks++
	return r.parsePrimitiveBlock(data)
}

// readBlob reads one length-prefixed blob and returns its type and decompressed
// contents. The returned slice is owned by the Reader.
func (r *Reader) readBlob() (string, []byte, error) {
	start := r.consumed
	r.curBlobOff = start
	var lenBuf [4]byte
	if _, err := io.ReadFull(r.br, lenBuf[:]); err != nil {
		return "", nil, err
	}
	r.consumed += 4
	n := int(binary.BigEndian.Uint32(lenBuf[:]))
	if n <= 0 || n > MaxBlobHeaderSize {
		return "", nil, fmt.Errorf("%w: blob header length %d", ErrCorrupt, n)
	}
	if cap(r.hdrBuf) < n {
		r.hdrBuf = make([]byte, n)
	}
	hb := r.hdrBuf[:n]
	if _, err := io.ReadFull(r.br, hb); err != nil {
		return "", nil, unexpectedEOF(err)
	}
	r.consumed += int64(n)
	typ, size, err := parseBlobHeader(hb)
	if err != nil {
		return "", nil, err
	}
	if size <= 0 || size > r.maxBlob {
		return "", nil, fmt.Errorf("%w: compressed blob of %d bytes (limit %d)", ErrBlobTooLarge, size, r.maxBlob)
	}
	if cap(r.blobBuf) < size {
		r.blobBuf = make([]byte, size)
	}
	bb := r.blobBuf[:size]
	if _, err := io.ReadFull(r.br, bb); err != nil {
		return "", nil, unexpectedEOF(err)
	}
	r.consumed += int64(size)
	if size > r.MaxCompressedBlob {
		r.MaxCompressedBlob = size
	}
	data, err := r.decodeBlob(bb)
	if err != nil {
		return "", nil, err
	}
	if len(data) > r.MaxRawBlob {
		r.MaxRawBlob = len(data)
	}
	return typ, data, nil
}

func unexpectedEOF(err error) error {
	if err == io.EOF {
		return io.ErrUnexpectedEOF
	}
	return err
}

func parseBlobHeader(b []byte) (string, int, error) {
	var typ string
	size := -1
	it := fieldIter{b: b}
	for {
		num, wire, u, sub, ok, err := it.next()
		if err != nil {
			return "", 0, err
		}
		if !ok {
			break
		}
		switch {
		case num == 1 && wire == 2:
			typ = string(sub)
		case num == 3 && wire == 0:
			size = int(u)
		}
	}
	if typ == "" || size < 0 {
		return "", 0, fmt.Errorf("%w: incomplete BlobHeader", ErrCorrupt)
	}
	return typ, size, nil
}

func (r *Reader) decodeBlob(b []byte) ([]byte, error) {
	var raw, zdata []byte
	rawSize := -1
	it := fieldIter{b: b}
	for {
		num, wire, u, sub, ok, err := it.next()
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		switch {
		case num == 1 && wire == 2:
			raw = sub
		case num == 2 && wire == 0:
			rawSize = int(u)
		case num == 3 && wire == 2:
			zdata = sub
		case num >= 4 && num <= 7:
			return nil, fmt.Errorf("%w: unsupported blob compression (field %d); only raw and zlib are supported", ErrCorrupt, num)
		}
	}
	if raw != nil {
		if len(raw) > r.maxBlob {
			return nil, fmt.Errorf("%w: raw blob of %d bytes (limit %d)", ErrBlobTooLarge, len(raw), r.maxBlob)
		}
		return raw, nil
	}
	if zdata == nil {
		return nil, fmt.Errorf("%w: empty Blob", ErrCorrupt)
	}
	if rawSize > r.maxBlob {
		return nil, fmt.Errorf("%w: blob declares %d uncompressed bytes (limit %d)", ErrBlobTooLarge, rawSize, r.maxBlob)
	}
	return r.inflate(zdata, rawSize)
}

func (r *Reader) inflate(src []byte, rawSize int) ([]byte, error) {
	r.zsrc.Reset(src)
	if r.zr == nil {
		zr, err := zlib.NewReader(&r.zsrc)
		if err != nil {
			return nil, fmt.Errorf("%w: zlib: %v", ErrCorrupt, err)
		}
		r.zr = zr
	} else if err := r.zr.(zlib.Resetter).Reset(&r.zsrc, nil); err != nil {
		return nil, fmt.Errorf("%w: zlib: %v", ErrCorrupt, err)
	}

	// Allow one byte of overshoot so a block of exactly maxBlob bytes is not
	// mistaken for a too-large one.
	limit := r.maxBlob + 1
	if rawSize > 0 && rawSize < limit {
		limit = rawSize
	}
	buf := r.dataBuf[:0]
	if cap(buf) > limit {
		buf = buf[:0:limit]
	}
	if cap(buf) == 0 {
		buf = make([]byte, 0, limit)
	}
	for {
		if len(buf) == cap(buf) {
			if cap(buf) >= limit {
				return nil, fmt.Errorf("%w: decompressed blob exceeds %d bytes", ErrBlobTooLarge, r.maxBlob)
			}
			newCap := cap(buf) * 2
			if newCap < 64<<10 {
				newCap = 64 << 10
			}
			if newCap > limit {
				newCap = limit
			}
			nb := make([]byte, len(buf), newCap)
			copy(nb, buf)
			buf = nb
		}
		n, err := r.zr.Read(buf[len(buf):cap(buf)])
		buf = buf[:len(buf)+n]
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("%w: zlib: %v", ErrCorrupt, err)
		}
	}
	if len(buf) > r.maxBlob {
		return nil, fmt.Errorf("%w: decompressed blob exceeds %d bytes", ErrBlobTooLarge, r.maxBlob)
	}
	r.dataBuf = buf
	return buf, nil
}

func (r *Reader) parseHeaderBlock(b []byte) error {
	it := fieldIter{b: b}
	for {
		num, wire, u, sub, ok, err := it.next()
		if err != nil {
			return err
		}
		if !ok {
			break
		}
		switch {
		case num == 1 && wire == 2:
			bb, err := parseHeaderBBox(sub)
			if err != nil {
				return err
			}
			r.hdr.BBox = &bb
		case num == 4 && wire == 2:
			r.hdr.RequiredFeatures = append(r.hdr.RequiredFeatures, string(sub))
		case num == 5 && wire == 2:
			r.hdr.OptionalFeatures = append(r.hdr.OptionalFeatures, string(sub))
		case num == 16 && wire == 2:
			r.hdr.WritingProgram = string(sub)
		case num == 17 && wire == 2:
			r.hdr.Source = string(sub)
		case num == 32 && wire == 0:
			r.hdr.ReplicationTimestamp = int64(u)
		case num == 33 && wire == 0:
			r.hdr.ReplicationSequence = int64(u)
		case num == 34 && wire == 2:
			r.hdr.ReplicationURL = string(sub)
		}
	}
	for _, f := range r.hdr.RequiredFeatures {
		switch f {
		case "OsmSchema-V0.6", "DenseNodes":
		case "HistoricalInformation":
			return fmt.Errorf("%w: history files are not supported", ErrCorrupt)
		default:
			return fmt.Errorf("%w: unsupported required feature %q", ErrCorrupt, f)
		}
	}
	return nil
}

func parseHeaderBBox(b []byte) (BBox, error) {
	var bb BBox
	it := fieldIter{b: b}
	for {
		num, wire, u, _, ok, err := it.next()
		if err != nil {
			return bb, err
		}
		if !ok {
			break
		}
		if wire != 0 {
			continue
		}
		v := float64(zigzag(u)) * 1e-9
		switch num {
		case 1:
			bb.Left = v
		case 2:
			bb.Right = v
		case 3:
			bb.Top = v
		case 4:
			bb.Bottom = v
		}
	}
	return bb, nil
}

func (r *Reader) parsePrimitiveBlock(b []byte) error {
	r.groups = r.groups[:0]
	r.strings = r.strings[:0]
	r.objs = r.objs[:0]
	r.dense = r.dense[:0]
	r.next = 0

	gran, dateGran := int64(100), int64(1000)
	var latOff, lonOff int64
	var stData []byte

	it := fieldIter{b: b}
	for {
		num, wire, u, sub, ok, err := it.next()
		if err != nil {
			return err
		}
		if !ok {
			break
		}
		switch {
		case num == 1 && wire == 2:
			stData = sub
		case num == 2 && wire == 2:
			r.groups = append(r.groups, sub)
		case num == 17 && wire == 0:
			gran = int64(u)
		case num == 18 && wire == 0:
			dateGran = int64(u)
		case num == 19 && wire == 0:
			latOff = int64(u)
		case num == 20 && wire == 0:
			lonOff = int64(u)
		}
	}
	if stData == nil {
		return fmt.Errorf("%w: PrimitiveBlock without string table", ErrCorrupt)
	}
	if gran == 0 {
		gran = 100
	}
	if dateGran == 0 {
		dateGran = 1000
	}
	if err := r.parseStringTable(stData); err != nil {
		return err
	}
	r.gran, r.dateGran, r.latOff, r.lonOff = gran, dateGran, latOff, lonOff
	for _, g := range r.groups {
		if err := r.parseGroup(g); err != nil {
			return err
		}
	}
	return nil
}

func (r *Reader) parseStringTable(b []byte) error {
	it := fieldIter{b: b}
	for {
		num, wire, _, sub, ok, err := it.next()
		if err != nil {
			return err
		}
		if !ok {
			break
		}
		if num == 1 && wire == 2 {
			r.strings = append(r.strings, string(sub))
		}
	}
	return nil
}

func (r *Reader) parseGroup(b []byte) error {
	it := fieldIter{b: b}
	for {
		num, wire, _, sub, ok, err := it.next()
		if err != nil {
			return err
		}
		if !ok {
			break
		}
		if wire != 2 {
			continue
		}
		switch num {
		case 1:
			r.objs = append(r.objs, objRef{kind: Node, dense: -1, b: sub})
		case 2:
			if err := r.parseDense(sub); err != nil {
				return err
			}
		case 3:
			r.objs = append(r.objs, objRef{kind: Way, dense: -1, b: sub})
		case 4:
			r.objs = append(r.objs, objRef{kind: Relation, dense: -1, b: sub})
		}
	}
	return nil
}

func (r *Reader) parseDense(b []byte) error {
	var g denseGroup
	var infoSub []byte

	it := fieldIter{b: b}
	for {
		num, wire, _, sub, ok, err := it.next()
		if err != nil {
			return err
		}
		if !ok {
			break
		}
		if wire != 2 {
			continue
		}
		switch num {
		case 1:
			g.ids, err = appendPackedSint64Delta(g.ids[:0], sub)
		case 5:
			infoSub = sub
		case 8:
			g.lats, err = appendPackedSint64Delta(g.lats[:0], sub)
		case 9:
			g.lons, err = appendPackedSint64Delta(g.lons[:0], sub)
		case 10:
			g.kv, err = appendPackedUint64(g.kv[:0], sub)
		}
		if err != nil {
			return err
		}
	}
	if len(g.lats) != len(g.ids) || len(g.lons) != len(g.ids) {
		return fmt.Errorf("%w: inconsistent DenseNodes arrays", ErrCorrupt)
	}
	if infoSub != nil {
		if err := parseDenseInfo(&g, infoSub, r.dateGran); err != nil {
			return err
		}
	}
	if len(g.kv) > 0 {
		g.kvOff = make([]int, len(g.ids)+1)
		k := 0
		for i := range g.ids {
			g.kvOff[i] = k
			for k < len(g.kv) && g.kv[k] != 0 {
				k += 2
			}
			if k >= len(g.kv) {
				return fmt.Errorf("%w: unterminated DenseNodes tag list", ErrCorrupt)
			}
			k++ // skip terminator
		}
		g.kvOff[len(g.ids)] = k
	}

	gi := len(r.dense)
	r.dense = append(r.dense, g)
	for i := range g.ids {
		r.objs = append(r.objs, objRef{kind: Node, dense: gi, idx: i})
	}
	return nil
}

func parseDenseInfo(g *denseGroup, b []byte, dateGran int64) error {
	var deltas []int64
	it := fieldIter{b: b}
	for {
		num, wire, _, sub, ok, err := it.next()
		if err != nil {
			return err
		}
		if !ok {
			break
		}
		if wire != 2 {
			continue
		}
		switch num {
		case 1:
			g.vectors, err = appendPackedUint32(g.vectors[:0], sub)
		case 2:
			deltas, err = appendPackedSint64Delta(deltas[:0], sub)
		}
		if err != nil {
			return err
		}
	}
	g.stamps = make([]int64, len(deltas))
	var acc int64
	for i, d := range deltas {
		acc += d
		g.stamps[i] = acc * dateGran / 1000
	}
	return nil
}

func (r *Reader) decode(ref *objRef) (*Entry, error) {
	r.entry = Entry{Type: ref.kind, Tags: r.tags}
	clear(r.tags)

	switch ref.kind {
	case Node:
		if ref.dense >= 0 {
			return r.decodeDenseNode(ref.dense, ref.idx)
		}
		return r.decodeNode(ref.b)
	case Way:
		return r.decodeWay(ref.b)
	case Relation:
		return r.decodeRelation(ref.b)
	}
	return nil, fmt.Errorf("%w: unknown object type", ErrCorrupt)
}

func (r *Reader) decodeDenseNode(gi, i int) (*Entry, error) {
	g := &r.dense[gi]
	if i >= len(g.ids) || i >= len(g.lats) || i >= len(g.lons) {
		return nil, fmt.Errorf("%w: DenseNodes index out of range", ErrCorrupt)
	}
	e := &r.entry
	e.ID = g.ids[i]
	e.Lat = coord(g.lats[i], r.latOff, r.gran)
	e.Lon = coord(g.lons[i], r.lonOff, r.gran)
	if i < len(g.vectors) {
		e.Version = int32(g.vectors[i])
	}
	if i < len(g.stamps) {
		e.Timestamp = g.stamps[i]
	}
	if len(g.kvOff) > 0 {
		for k := g.kvOff[i]; k+1 < g.kvOff[i+1]; k += 2 {
			key, ok := r.str(g.kv[k])
			if !ok {
				return nil, fmt.Errorf("%w: tag key index out of range", ErrCorrupt)
			}
			val, ok := r.str(g.kv[k+1])
			if !ok {
				return nil, fmt.Errorf("%w: tag value index out of range", ErrCorrupt)
			}
			r.tags[key] = val
		}
	}
	return e, nil
}

func (r *Reader) decodeNode(b []byte) (*Entry, error) {
	e := &r.entry
	var id, lat, lon, ts int64
	var version int32
	var keys, vals []uint64

	it := fieldIter{b: b}
	for {
		num, wire, u, sub, ok, err := it.next()
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		switch {
		case num == 1 && wire == 0:
			id = zigzag(u)
		case num == 2 && wire == 2:
			keys, err = appendPackedUint64(r.keys[:0], sub)
			r.keys = keys
		case num == 3 && wire == 2:
			vals, err = appendPackedUint64(r.vals[:0], sub)
			r.vals = vals
		case num == 4 && wire == 2:
			version, ts, err = parseInfo(sub, r.dateGran)
		case num == 8 && wire == 0:
			lat = zigzag(u)
		case num == 9 && wire == 0:
			lon = zigzag(u)
		}
		if err != nil {
			return nil, err
		}
	}
	e.ID = id
	e.Version = version
	e.Timestamp = ts
	e.Lat = coord(lat, r.latOff, r.gran)
	e.Lon = coord(lon, r.lonOff, r.gran)
	return e, r.fillTags(keys, vals)
}

func (r *Reader) decodeWay(b []byte) (*Entry, error) {
	e := &r.entry
	var id int64
	var keys, vals []uint64
	var version int32
	var ts int64

	r.refs = r.refs[:0]
	r.refLats = r.refLats[:0]
	r.refLons = r.refLons[:0]

	it := fieldIter{b: b}
	for {
		num, wire, u, sub, ok, err := it.next()
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		switch {
		case num == 1 && wire == 0:
			id = int64(u)
		case num == 2 && wire == 2:
			keys, err = appendPackedUint64(r.keys[:0], sub)
			r.keys = keys
		case num == 3 && wire == 2:
			vals, err = appendPackedUint64(r.vals[:0], sub)
			r.vals = vals
		case num == 4 && wire == 2:
			version, ts, err = parseInfo(sub, r.dateGran)
		case num == 8 && wire == 2:
			r.refs, err = appendPackedSint64Delta(r.refs, sub)
		case num == 9 && wire == 2:
			r.refLats, err = appendPackedSint64DeltaFloat(r.refLats, sub, r.latOff, r.gran)
		case num == 10 && wire == 2:
			r.refLons, err = appendPackedSint64DeltaFloat(r.refLons, sub, r.lonOff, r.gran)
		}
		if err != nil {
			return nil, err
		}
	}
	e.ID = id
	e.Version = version
	e.Timestamp = ts
	e.Refs = r.refs
	if len(r.refLats) > 0 && len(r.refLons) > 0 {
		e.RefLats = r.refLats
		e.RefLons = r.refLons
	}
	return e, r.fillTags(keys, vals)
}

func (r *Reader) decodeRelation(b []byte) (*Entry, error) {
	e := &r.entry
	var id int64
	var keys, vals []uint64
	var version int32
	var ts int64

	r.refs = r.refs[:0]
	r.members = r.members[:0]
	var roles, types []uint64

	it := fieldIter{b: b}
	for {
		num, wire, u, sub, ok, err := it.next()
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		switch {
		case num == 1 && wire == 0:
			id = int64(u)
		case num == 2 && wire == 2:
			keys, err = appendPackedUint64(r.keys[:0], sub)
			r.keys = keys
		case num == 3 && wire == 2:
			vals, err = appendPackedUint64(r.vals[:0], sub)
			r.vals = vals
		case num == 4 && wire == 2:
			version, ts, err = parseInfo(sub, r.dateGran)
		case num == 8 && wire == 2:
			roles, err = appendPackedUint64(roles[:0], sub)
		case num == 9 && wire == 2:
			r.refs, err = appendPackedSint64Delta(r.refs, sub)
		case num == 10 && wire == 2:
			types, err = appendPackedUint64(types[:0], sub)
		}
		if err != nil {
			return nil, err
		}
	}
	if len(roles) != len(r.refs) || len(types) != len(r.refs) {
		return nil, fmt.Errorf("%w: inconsistent relation member arrays", ErrCorrupt)
	}
	for i, ref := range r.refs {
		if types[i] > 2 {
			return nil, fmt.Errorf("%w: relation member type %d", ErrCorrupt, types[i])
		}
		role, ok := r.str(roles[i])
		if !ok {
			return nil, fmt.Errorf("%w: member role index out of range", ErrCorrupt)
		}
		r.members = append(r.members, Member{
			Type: Kind(types[i] + 1),
			Ref:  ref,
			Role: role,
		})
	}
	e.ID = id
	e.Version = version
	e.Timestamp = ts
	e.Members = r.members
	return e, r.fillTags(keys, vals)
}

func (r *Reader) fillTags(keys, vals []uint64) error {
	if len(keys) != len(vals) {
		return fmt.Errorf("%w: tag key/value count mismatch", ErrCorrupt)
	}
	for i := range keys {
		k, ok := r.str(keys[i])
		if !ok {
			return fmt.Errorf("%w: tag key index out of range", ErrCorrupt)
		}
		v, ok := r.str(vals[i])
		if !ok {
			return fmt.Errorf("%w: tag value index out of range", ErrCorrupt)
		}
		r.tags[k] = v
	}
	return nil
}

func (r *Reader) str(i uint64) (string, bool) {
	if i >= uint64(len(r.strings)) {
		return "", false
	}
	return r.strings[i], true
}

func parseInfo(b []byte, dateGran int64) (int32, int64, error) {
	var version int32
	var ts int64
	it := fieldIter{b: b}
	for {
		num, wire, u, _, ok, err := it.next()
		if err != nil {
			return 0, 0, err
		}
		if !ok {
			break
		}
		if wire != 0 {
			continue
		}
		switch num {
		case 1:
			version = int32(u)
		case 2:
			ts = int64(u) * dateGran / 1000
		}
	}
	return version, ts, nil
}

func coord(v, off, gran int64) float64 {
	return 1e-9 * float64(off+gran*v)
}

// fieldIter walks the fields of a protobuf message.
type fieldIter struct {
	b   []byte
	pos int
}

// next returns the next field. ok is false at the end of the message.
// For wire type 0, u holds the value. For wire type 2, sub holds the bytes.
func (f *fieldIter) next() (num, wire int, u uint64, sub []byte, ok bool, err error) {
	if f.pos >= len(f.b) {
		return 0, 0, 0, nil, false, nil
	}
	key, n := binary.Uvarint(f.b[f.pos:])
	if n <= 0 {
		return 0, 0, 0, nil, false, fmt.Errorf("%w: bad field key", ErrCorrupt)
	}
	f.pos += n
	num, wire = int(key>>3), int(key&7)
	switch wire {
	case 0:
		v, m := binary.Uvarint(f.b[f.pos:])
		if m <= 0 {
			return 0, 0, 0, nil, false, ErrTruncated
		}
		f.pos += m
		u = v
	case 1:
		if len(f.b)-f.pos < 8 {
			return 0, 0, 0, nil, false, ErrTruncated
		}
		f.pos += 8
	case 2:
		l, m := binary.Uvarint(f.b[f.pos:])
		if m <= 0 {
			return 0, 0, 0, nil, false, ErrTruncated
		}
		f.pos += m
		if l > uint64(len(f.b)-f.pos) {
			return 0, 0, 0, nil, false, ErrTruncated
		}
		sub = f.b[f.pos : f.pos+int(l)]
		f.pos += int(l)
	case 5:
		if len(f.b)-f.pos < 4 {
			return 0, 0, 0, nil, false, ErrTruncated
		}
		f.pos += 4
	default:
		return 0, 0, 0, nil, false, fmt.Errorf("%w: unsupported wire type %d", ErrCorrupt, wire)
	}
	return num, wire, u, sub, true, nil
}

func zigzag(u uint64) int64 {
	return int64(u>>1) ^ -int64(u&1)
}

func appendPackedUint64(dst []uint64, b []byte) ([]uint64, error) {
	for len(b) > 0 {
		v, n := binary.Uvarint(b)
		if n <= 0 {
			return dst, ErrTruncated
		}
		dst = append(dst, v)
		b = b[n:]
	}
	return dst, nil
}

func appendPackedUint32(dst []uint32, b []byte) ([]uint32, error) {
	for len(b) > 0 {
		v, n := binary.Uvarint(b)
		if n <= 0 {
			return dst, ErrTruncated
		}
		dst = append(dst, uint32(v))
		b = b[n:]
	}
	return dst, nil
}

// appendPackedSint64Delta decodes a packed, delta-coded sint64 array, as used
// by DenseNodes and Way.refs.
func appendPackedSint64Delta(dst []int64, b []byte) ([]int64, error) {
	var last int64
	for len(b) > 0 {
		v, n := binary.Uvarint(b)
		if n <= 0 {
			return dst, ErrTruncated
		}
		last += zigzag(v)
		dst = append(dst, last)
		b = b[n:]
	}
	return dst, nil
}

// appendPackedSint64DeltaFloat decodes delta-coded nano-degree values straight
// into degrees.
func appendPackedSint64DeltaFloat(dst []float64, b []byte, off, gran int64) ([]float64, error) {
	var last int64
	for len(b) > 0 {
		v, n := binary.Uvarint(b)
		if n <= 0 {
			return dst, ErrTruncated
		}
		last += zigzag(v)
		dst = append(dst, coord(last, off, gran))
		b = b[n:]
	}
	return dst, nil
}
