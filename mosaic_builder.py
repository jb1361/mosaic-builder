#!/usr/bin/env python3

try: 	
	import os, sys, math, argparse, random, json, time
	from PIL import Image
	import numpy as np
except Exception as e:
	print(f"Error during import: {e}")

def avg_color(img: Image.Image):
	small = img.resize((16, 16), Image.BOX)
	arr = np.array(small, dtype=np.float32)
	return tuple(float(x) for x in np.mean(arr.reshape(-1, 3), axis=0))

def parse_args():
	p = argparse.ArgumentParser()
	# generation params
	p.add_argument("--base", help="Path to base image")
	p.add_argument("--tiles", help="Directory containing tile images")
	p.add_argument("--out", help="Output directory for chunks and manifest")
	p.add_argument("--grid-width", type=int, default=50, help="Tiles across")
	p.add_argument("--grid-height", type=int, default=50, help="Tiles across")
	p.add_argument("--tile-width", type=int, default=500, help="Pixel width of each tile canvas")
	p.add_argument("--chunk-w", type=int, default=5, help="Tiles per chunk horizontally")
	p.add_argument("--chunk-h", type=int, default=5, help="Tiles per chunk vertically")
	p.add_argument("--color-match", action="store_true", help="Nearest-avg-color tile selection")
	p.add_argument("--seed", type=int, default=1337, help="Random seed (when not color-matching)")
	# stitching params
	p.add_argument("--stitch-out", help="Output path for stitched full mosaic (e.g., mosaic_out/full.png)")
	p.add_argument("--stitch-from", help="Path to an existing manifest.json to stitch from")
	p.add_argument("--stitch-only", action="store_true", help="Only stitch using --stitch-from; skip generation")
	return p.parse_args()

def load_or_fail(path: str):
	if not path:
		print("Missing required path.", file=sys.stderr); sys.exit(1)

def save_image(img: Image.Image, path: str, retries: int = 10, **kwargs):
	# On Windows, overwriting a file that another process has memory-mapped (e.g. Explorer's
	# thumbnail generator, "COM Surrogate") fails with OSError 22/13. Retry, since those
	# locks are usually brief, then fail with a useful message.
	for attempt in range(retries):
		try:
			img.save(path, **kwargs)
			return
		except OSError as e:
			if e.errno not in (13, 22) or attempt == retries - 1:
				raise OSError(e.errno, f"Could not write {path}. Another program may have it open "
					"(close any Explorer window/image viewer showing the output folder, or delete "
					"the old output files first)") from e
			time.sleep(0.5 * (attempt + 1))

def gen_chunks(args):
	random.seed(args.seed)
	os.makedirs(args.out, exist_ok=True)

	base = Image.open(args.base).convert("RGB")
	bw, bh = base.size
	grid_w = args.grid_width
	grid_h = max(1, int((bh / bw) * grid_w))
	#grid_h = args.grid_height
	base_small = base.resize((grid_w, grid_h), Image.LANCZOS)

	valid_ext = (".png", ".jpg", ".jpeg", ".webp")
	tile_files = [os.path.join(args.tiles, f) for f in os.listdir(args.tiles) if f.lower().endswith(valid_ext)]
	if not tile_files:
		print("No tile images found.", file=sys.stderr); sys.exit(1)
	random.shuffle(tile_files)

	with Image.open(tile_files[0]).convert("RGB") as probe:
		ratio = probe.height / probe.width if probe.width else 1.0
	tile_w = args.tile_width
	tile_h = max(64, int(tile_w * ratio))

	tile_cache = {}  # path -> (canvas_img, avg_color)

	def get_cached_tile(path: str):
		if path in tile_cache:
			return tile_cache[path]
		try:
			with Image.open(path).convert("RGB") as im:
				scale = min(tile_w / im.width, tile_h / im.height)
				new_w = max(1, int(im.width * scale))
				new_h = max(1, int(im.height * scale))
				im_resized = im.resize((new_w, new_h), Image.LANCZOS)
				canvas = Image.new("RGB", (tile_w, tile_h), (0, 0, 0))
				off_x = (tile_w - new_w) // 2
				off_y = (tile_h - new_h) // 2
				canvas.paste(im_resized, (off_x, off_y))
				ac = avg_color(canvas)
				tile_cache[path] = (canvas, ac)
				return tile_cache[path]
		except Exception:
			canvas = Image.new("RGB", (tile_w, tile_h), (0, 0, 0))
			ac = (0.0, 0.0, 0.0)
			tile_cache[path] = (canvas, ac)
			return tile_cache[path]

	def nearest_tile(target_rgb):
		tr, tg, tb = target_rgb
		best_canvas = None
		best_d = 1e18
		for p in tile_files:
			canvas, ac = get_cached_tile(p)
			dr = tr - ac[0]; dg = tg - ac[1]; db = tb - ac[2]
			d = dr*dr + dg*dg + db*db
			if d < best_d:
				best_d = d
				best_canvas = canvas
		return best_canvas

	chunk_w, chunk_h = args.chunk_w, args.chunk_h
	col_chunks = math.ceil(grid_w / chunk_w)
	row_chunks = math.ceil(grid_h / chunk_h)

	manifest = {
		"base": os.path.abspath(args.base),
		"tiles_dir": os.path.abspath(args.tiles),
		"out_dir": os.path.abspath(args.out),
		"grid_size": [grid_w, grid_h],
		"tile_size": [tile_w, tile_h],
		"chunk_tile_size": [chunk_w, chunk_h],
		"color_match": bool(args.color_match),
		"chunks": []
	}

	for r in range(row_chunks):
		cy = r * chunk_h
		for c in range(col_chunks):
			cx = c * chunk_w
			chunk_img = Image.new("RGB", (chunk_w * tile_w, chunk_h * tile_h), (0,0,0))
			for y in range(chunk_h):
				gy = cy + y
				if gy >= grid_h: continue
				for x in range(chunk_w):
					gx = cx + x
					if gx >= grid_w: continue
					if args.color_match:
						target = base_small.getpixel((gx, gy))
						tile_canvas = nearest_tile(target)
					else:
						p = tile_files[(gy * grid_w + gx) % len(tile_files)]
						tile_canvas, _ = get_cached_tile(p)
					chunk_img.paste(tile_canvas, (x * tile_w, y * tile_h))
			out_name = f"chunk_r{cy}_c{cx}.png"
			out_path = os.path.join(args.out, out_name)
			save_image(chunk_img, out_path, optimize=True)
			manifest["chunks"].append(out_name)
			print(f"Wrote {out_path}")

	manifest_path = os.path.join(args.out, "manifest.json")
	with open(manifest_path, "w") as f:
		json.dump(manifest, f, indent=2)
	print("Done. Manifest saved at", manifest_path)
	return manifest_path

def stitch_from_manifest(manifest_path: str, stitch_out: str):
	with open(manifest_path, "r") as f:
		manifest = json.load(f)

	out_dir = manifest["out_dir"]
	grid_w, grid_h = manifest["grid_size"]
	tile_w, tile_h = manifest["tile_size"]
	chunk_w, chunk_h = manifest["chunk_tile_size"]
	chunks = manifest["chunks"]

	big_w = grid_w * tile_w
	big_h = grid_h * tile_h
	print(f"Stitching {len(chunks)} chunks into {big_w}x{big_h} ...")
	canvas = Image.new("RGB", (big_w, big_h), (0,0,0))

	for name in chunks:
		try:
			base = os.path.splitext(os.path.basename(name))[0]
			parts = base.split("_")
			cy = int(parts[1][1:])  # r{cy}
			cx = int(parts[2][1:])  # c{cx}
		except Exception:
			print(f"Skipping unparsable chunk name: {name}", file=sys.stderr)
			continue
		x_px = cx * tile_w
		y_px = cy * tile_h
		p = os.path.join(out_dir, name)
		with Image.open(p).convert("RGB") as im:
			canvas.paste(im, (x_px, y_px))

	os.makedirs(os.path.dirname(stitch_out) or ".", exist_ok=True)
	save_image(canvas, stitch_out, optimize=True)
	print(f"Stitched mosaic saved to {stitch_out}")

def main():
	print('running')
	args = parse_args()
	if args.stitch_only:
		load_or_fail(args.stitch_from)
		load_or_fail(args.stitch_out)
		stitch_from_manifest(args.stitch_from, args.stitch_out)
		return
	if not args.base or not args.tiles or not args.out:
		print("For generation, provide --base, --tiles, and --out. Or use --stitch-only.", file=sys.stderr)
		sys.exit(1)
	manifest_path = gen_chunks(args)
	if args.stitch_out:
		stitch_from_manifest(manifest_path, args.stitch_out)


main()
input("Press enter to exit;")