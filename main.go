// Test harness for the index directory: pick a query tile by (tile size in
// degrees, x, y indices) and print every entry whose stored bounding box
// intersects it.
//
// The index is built automatically if <base>/<md5 of the pbf> does not exist.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	pbf "schoolmap/src"
)

func main() {
	var (
		base       = flag.String("index", "/tmp/schoolmap", "base directory for index dirs")
		pbfPath    = flag.String("pbf", "assets/japan-261116.osm.pbf", "source PBF")
		tileW      = flag.Float64("tile-w", 8, "gigantic tile width in degrees")
		tileH      = flag.Float64("tile-h", 8, "gigantic tile height in degrees")
		maxOverlap = flag.Int("max-overlap", 2, "max tiles an entry is copied into")
		jobs       = flag.Int("jobs", 2, "parallel tile builders")
		force      = flag.Bool("force", false, "rebuild the index even if it exists")

		tw     = flag.Float64("tw", 0.01, "query tile width in degrees")
		th     = flag.Float64("th", 0.01, "query tile height in degrees")
		tx     = flag.Int("x", 0, "query tile column index (origin = index bounds min corner)")
		ty     = flag.Int("y", 0, "query tile row index")
		limit  = flag.Int("limit", 0, "max entries to print (0 = all)")
		boxes  = flag.Bool("boxes", false, "print each entry's stored bounding box")
		verify = flag.Bool("verify", false, "compare hits against a full PBF scan")
		pngOut = flag.String("png", "", "render the query tile to a PNG at this path")
		pxSize = flag.Int("px", 1024, "PNG size in pixels per side")
		noOut  = flag.Bool("no-outline", false, "omit the red query-tile outline")
		poi    = flag.Bool("poi", false, "draw amenity/shop/leisure node dots")
	)
	flag.Parse()

	start := time.Now()
	dir, _, err := pbf.BuildIndex(*base, *pbfPath, pbf.IndexOpts{
		TileW: *tileW, TileH: *tileH, MaxOverlap: *maxOverlap, Jobs: *jobs, Force: *force,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("index: %s\n", dir)
	idx, err := pbf.OpenIndex(dir)
	if err != nil {
		log.Fatal(err)
	}
	defer idx.Close()
	m := idx.Meta()
	b := m.Bounds
	fmt.Printf("index: %d entries (%d nodes, %d ways, %d rels), %d tiles, %d oversized, built in %s\n",
		m.Total, m.Nodes, m.Ways, m.Rels, len(m.Tiles), m.Oversized[1]+m.Oversized[2]+m.Oversized[3],
		time.Since(start).Round(time.Millisecond))

	q := [4]float64{
		b[0] + float64(*tx)*(*tw),
		b[1] + float64(*ty)*(*th),
		b[0] + float64(*tx+1)*(*tw),
		b[1] + float64(*ty+1)*(*th),
	}
	fmt.Printf("query: x=%d y=%d size=%.4fx%.4f deg -> %.6f,%.6f .. %.6f,%.6f\n",
		*tx, *ty, *tw, *th, q[0], q[1], q[2], q[3])

	var counts [4]int
	overCount := 0
	printed := 0
	got := map[uint64]struct{}{}
	st, err := idx.Search(q[0], q[1], q[2], q[3], func(k pbf.Kind, id int64, box [4]float64, loc pbf.BlobRef, over bool) bool {
		counts[k]++
		if over {
			overCount++
		}
		if *verify {
			got[uint64(k)<<40|uint64(id)] = struct{}{}
		}
		if *limit == 0 || printed < *limit {
			if *boxes {
				fmt.Printf("%s %d [%.6f,%.6f .. %.6f,%.6f]%s\n", k, id, box[0], box[1], box[2], box[3],
					map[bool]string{true: " oversized", false: ""}[over])
			} else {
				fmt.Printf("%s %d%s\n", k, id, map[bool]string{true: " oversized", false: ""}[over])
			}
			printed++
		}
		return true
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("hits: %d nodes, %d ways, %d relations (%d total, %d oversized)\n",
		counts[pbf.Node], counts[pbf.Way], counts[pbf.Relation], st.Hits, overCount)
	fmt.Printf("read: %d tree nodes, %.1f KB of tile files\n", st.NodesRead, float64(st.BytesRead)/1e3)

	if *verify {
		verifyAgainstPBF(*pbfPath, q, got)
	}

	if *pngOut != "" {
		start := time.Now()
		data, err := idx.RenderTileStyled(pbf.TileRef{W: *tw, H: *th, X: *tx, Y: *ty}, *pbfPath, *pxSize,
			pbf.RenderStyle{NoOutline: *noOut, PoiDots: *poi})
		if err != nil {
			log.Fatal(err)
		}
		if err := os.WriteFile(*pngOut, data, 0o644); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("png: %s (%.1f KB) in %s\n", *pngOut, float64(len(data))/1e3, time.Since(start).Round(time.Millisecond))
	}
}

func verifyAgainstPBF(pbfPath string, q [4]float64, got map[uint64]struct{}) {
	want := map[uint64]struct{}{}
	err := pbf.ForEachEntryBox(pbfPath, func(k pbf.Kind, id int64, b [4]float64, _ pbf.BlobRef) error {
		if b[2] >= q[0] && b[0] <= q[2] && b[3] >= q[1] && b[1] <= q[3] {
			want[uint64(k)<<40|uint64(id)] = struct{}{}
		}
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}

	missing, extra := 0, 0
	for k := range want {
		if _, ok := got[k]; !ok {
			if missing < 10 {
				fmt.Printf("  missing %s %d\n", pbf.Kind(k>>40), int64(k&((1<<40)-1)))
			}
			missing++
		}
	}
	for k := range got {
		if _, ok := want[k]; !ok {
			if extra < 10 {
				fmt.Printf("  extra   %s %d\n", pbf.Kind(k>>40), int64(k&((1<<40)-1)))
			}
			extra++
		}
	}
	// The tree widens query boxes by boxEps for float32 safety, so a handful
	// of extra candidates just past the tile edge is expected.
	if missing != 0 || extra > len(want)/100+100 {
		fmt.Printf("VERIFY FAILED: %d missing, %d extra (want %d, got %d)\n",
			missing, extra, len(want), len(got))
		os.Exit(1)
	}
	fmt.Printf("verify: OK, all %d entries found (%d extra candidates from box rounding)\n",
		len(want), extra)
}
