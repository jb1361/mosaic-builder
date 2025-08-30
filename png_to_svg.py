#!/usr/bin/env python3
# tabs-only indentation

import sys, os, argparse, base64, io, math
from xml.sax.saxutils import escape

def write_svg_embed(png_path, svg_path):
	from PIL import Image
	with open(png_path, "rb") as f:
		data = f.read()
	img = Image.open(io.BytesIO(data))
	w, h = img.size
	b64 = base64.b64encode(data).decode("ascii")
	svg = []
	svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
	svg.append(f'<image href="data:image/png;base64,{b64}" x="0" y="0" width="{w}" height="{h}" />')
	svg.append('</svg>')
	with open(svg_path, "w", encoding="utf-8") as f:
		f.write("\n".join(svg))

def _rgb_to_hex(bgr_tuple):
	# OpenCV gives BGR; convert to hex RGB
	b, g, r = [int(max(0, min(255, v))) for v in bgr_tuple]
	return f'#{r:02x}{g:02x}{b:02x}'

def _contour_to_path_d(cnt):
	# Build "M x y L x y ... Z"
	points = cnt.reshape(-1, 2).tolist()
	if not points: return ""
	d = f"M {points[0][0]} {points[0][1]}"
	for x, y in points[1:]:
		d += f" L {x} {y}"
	d += " Z"
	return d

def write_svg_trace(png_path, svg_path, colors=8, blur=1, min_area=50, alpha_thr=1, simplify=0.005, downscale=1.0):
	import cv2
	import numpy as np

	# Load with alpha
	img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
	if img is None:
		raise RuntimeError("Failed to read image. Check path.")
	h, w = img.shape[:2]

	# Optional downscale (for speed)
	if downscale < 1.0:
		new_w = max(1, int(w * downscale))
		new_h = max(1, int(h * downscale))
		img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
		h, w = img.shape[:2]

	# Split channels
	if img.shape[2] == 4:
		bgr = img[:, :, :3]
		alpha = img[:, :, 3]
	else:
		bgr = img[:, :, :3]
		alpha = (255 * (bgr.sum(axis=2) > 0)).astype("uint8")

	# Mask of visible pixels
	vis_mask = (alpha >= alpha_thr).astype("uint8") * 255

	# K-means color quantization on visible pixels only
	flat = bgr.reshape(-1, 3)
	flat_mask = vis_mask.reshape(-1) > 0
	pts = flat[flat_mask]
	if pts.size == 0:
		# No visible pixels; just create blank SVG
		with open(svg_path, "w", encoding="utf-8") as f:
			f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"></svg>')
		return

	import numpy as np
	pts32 = np.float32(pts)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
	# cap colors to number of distinct pixels if tiny
	k = max(1, min(colors, len(pts32)))
	compactness, labels, centers = cv2.kmeans(pts32, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

	# Build a label image (default background -1)
	label_img = -1 * np.ones((h * w,), dtype=np.int32)
	label_img[flat_mask] = labels.flatten()
	label_img = label_img.reshape((h, w))

	# Optional blur to smooth masks
	if blur > 1:
		vis_mask = cv2.medianBlur(vis_mask, int(blur) | 1)

	# Create SVG
	paths = []
	paths.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
	paths.append('<g shape-rendering="geometricPrecision" fill-rule="evenodd">')

	for ci, center in enumerate(centers):
		# Mask for this color
		mask = (label_img == ci).astype("uint8") * 255
		if blur > 1:
			mask = cv2.medianBlur(mask, int(blur) | 1)

		# Clean tiny specks
		kernel = np.ones((3,3), np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

		# Find contours
		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		fill = _rgb_to_hex(center)
		for cnt in contours:
			area = cv2.contourArea(cnt)
			if area < float(min_area):
				continue
			# Simplify contour
			eps = max(1.0, simplify * cv2.arcLength(cnt, True))  # pixels
			approx = cv2.approxPolyDP(cnt, eps, True)
			d = _contour_to_path_d(approx)
			if d:
				paths.append(f'<path d="{escape(d)}" fill="{fill}" stroke="none"/>')

	paths.append('</g>')
	paths.append('</svg>')

	with open(svg_path, "w", encoding="utf-8") as f:
		f.write("\n".join(paths))

def main():
	parser = argparse.ArgumentParser(description="Convert PNG to SVG (embed or trace). Tabs-only indentation script.")
	parser.add_argument("--input", required=True, help="Input PNG path")
	parser.add_argument("--output", required=True, help="Output SVG path")
	parser.add_argument("--mode", choices=["embed","trace"], default="embed", help="embed: wrap PNG in SVG; trace: vectorize")
	# trace options
	parser.add_argument("--colors", type=int, default=8, help="Number of colors for vectorization (trace)")
	parser.add_argument("--blur", type=int, default=1, help="Median blur kernel (odd). 1 disables.")
	parser.add_argument("--min-area", type=float, default=50.0, help="Ignore contours smaller than this area (pixels)")
	parser.add_argument("--alpha", type=int, default=1, help="Alpha threshold (0-255) for visible pixels")
	parser.add_argument("--simplify", type=float, default=0.005, help="Polygon simplify factor (fraction of perimeter)")
	parser.add_argument("--downscale", type=float, default=1.0, help="Downscale factor (0<d<=1) before tracing")
	args = parser.parse_args()

	if args.mode == "embed":
		try:
			write_svg_embed(args.input, args.output)
		except ImportError:
			print("Pillow not installed. Run: pip install pillow", file=sys.stderr)
			sys.exit(1)
	else:
		try:
			write_svg_trace(
				args.input,
				args.output,
				colors=args.colors,
				blur=args.blur,
				min_area=args.min_area,
				alpha_thr=args.alpha,
				simplify=args.simplify,
				downscale=args.downscale
			)
		except ImportError as e:
			print("Tracing requires OpenCV and NumPy. Run: pip install opencv-python numpy", file=sys.stderr)
			sys.exit(1)

if __name__ == "__main__":
	main()
