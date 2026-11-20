package pbf

// entryfetch.go — random access to individual entries in a PBF file via the
// blob offsets recorded in Entry.Loc / lookup.bin.
//
// Fetching one entry costs one pread sequence (blob framing + compressed
// blob) plus one zlib decode (~1-3 ms). Because most entries of a query
// share blocks, callers should batch: group the BlobRefs by Off and pull
// every needed index from each block.
//
// No cross-call block caching here: every Block call decodes fresh, and the
// returned view is only valid until the next Block call (it shares the
// fetcher's buffers).

import (
	"encoding/binary"
	"fmt"
	"os"
)

// EntryFetcher reads individual entries from a PBF file by BlobRef. It holds
// one persistent file handle and reads with pread (ReadAt) — no seeks, no
// bufio. Not safe for concurrent use.
type EntryFetcher struct {
	f   *os.File
	r   *Reader // reused per block; buffers belong to the returned view
	hdr []byte  // reusable blob-header buffer
}

// NewEntryFetcher opens a PBF for random access.
func NewEntryFetcher(pbfPath string) (*EntryFetcher, error) {
	f, err := os.Open(pbfPath)
	if err != nil {
		return nil, err
	}
	r := &Reader{
		maxBlob: MaxBlobSize,
		tags:    make(map[string]string, 16),
	}
	return &EntryFetcher{f: f, r: r}, nil
}

// Close closes the file.
func (ef *EntryFetcher) Close() error { return ef.f.Close() }

// Block decodes the block at blob offset off (the offset of the blob's
// 4-byte length prefix) and returns a view of it. The view is valid until
// the next Block call.
func (ef *EntryFetcher) Block(off int64) (*BlockView, error) {
	// Blob framing: 4-byte big-endian header length, header, blob.
	var lenBuf [4]byte
	if _, err := ef.f.ReadAt(lenBuf[:], off); err != nil {
		return nil, err
	}
	n := int(binary.BigEndian.Uint32(lenBuf[:]))
	if n <= 0 || n > MaxBlobHeaderSize {
		return nil, fmt.Errorf("%w: blob header length %d at %d", ErrBadIndex, n, off)
	}
	if cap(ef.hdr) < n {
		ef.hdr = make([]byte, n)
	}
	if _, err := ef.f.ReadAt(ef.hdr[:n], off+4); err != nil {
		return nil, err
	}
	typ, size, err := parseBlobHeader(ef.hdr[:n])
	if err != nil {
		return nil, err
	}
	if typ != "OSMData" {
		return nil, fmt.Errorf("%w: blob at %d is %s", ErrBadIndex, off, typ)
	}
	blob := make([]byte, size)
	if _, err := ef.f.ReadAt(blob, off+4+int64(n)); err != nil {
		return nil, err
	}

	data, err := ef.r.decodeBlob(blob)
	if err != nil {
		return nil, err
	}
	if err := ef.r.parsePrimitiveBlock(data); err != nil {
		return nil, err
	}
	return &BlockView{r: ef.r}, nil
}

// BlockView gives access to the entries of one decoded block. All returned
// Entries share the fetcher's buffers and are only valid until the next
// Block or At call.
type BlockView struct {
	r *Reader
}

// At decodes the entry at position idx within the block.
func (v *BlockView) At(idx int32) (*Entry, error) {
	if idx < 0 || int(idx) >= len(v.r.objs) {
		return nil, fmt.Errorf("%w: entry index %d out of range (block has %d objects)",
			ErrBadIndex, idx, len(v.r.objs))
	}
	return v.r.decode(&v.r.objs[idx])
}

// Len returns the number of objects in the block.
func (v *BlockView) Len() int { return len(v.r.objs) }
