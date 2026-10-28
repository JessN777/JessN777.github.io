"""CS180 Project 4A. Coordinates are (x,y), at pixel centers."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image, ImageOps


def transform(points, H):
    p = np.c_[np.asarray(points, dtype=float), np.ones(len(points))] @ H.T
    if np.any(np.abs(p[:, 2]) < 1e-12):
        raise ValueError('A point maps to infinity.')
    return p[:, :2] / p[:, 2:]


def _normalize(p):
    center = p.mean(axis=0)
    distance = np.linalg.norm(p - center, axis=1).mean()
    if distance < 1e-12:
        raise ValueError('Correspondences must be distinct.')
    s = np.sqrt(2) / distance
    T = np.array([[s, 0, -s*center[0]], [0, s, -s*center[1]], [0, 0, 1]])
    return transform(p, T), T


def linear_system(source, destination):
    x, y = np.asarray(source).T
    u, v = np.asarray(destination).T
    A = np.zeros((2*len(x), 8))
    A[::2] = np.c_[x, y, np.ones(len(x)), np.zeros((len(x), 3)), -u*x, -u*y]
    A[1::2] = np.c_[np.zeros((len(x), 3)), x, y, np.ones(len(x)), -v*x, -v*y]
    return A, np.asarray(destination).reshape(-1)


def computeH(im1_pts, im2_pts):
    """Recover H mapping im1 to im2 using normalized 2N-by-8 least squares."""
    a, b = np.asarray(im1_pts, float), np.asarray(im2_pts, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2 or len(a) < 4:
        raise ValueError('Provide matching N-by-2 arrays with N >= 4.')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Coordinates must be finite.')
    a, T1 = _normalize(a)
    b, T2 = _normalize(b)
    A, rhs = linear_system(a, b)
    h, _, rank, _ = np.linalg.lstsq(A, rhs, rcond=None)
    if rank < 8:
        raise ValueError('Degenerate points: choose points spread across the image.')
    H = np.linalg.inv(T2) @ np.r_[h, 1].reshape(3, 3) @ T1
    scale = H[2, 2] if abs(H[2, 2]) > 1e-12 else np.linalg.norm(H)
    return H / scale


def bounds(image, H):
    h, w = image.shape[:2]
    corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    denominators = np.c_[corners, np.ones(4)] @ H[2]
    if denominators.min() <= 0 <= denominators.max():
        raise ValueError('The image crosses the projective horizon; choose another reference.')
    p = transform(corners, H)
    return (*np.floor(p.min(axis=0)).astype(int), *np.ceil(p.max(axis=0)).astype(int))


def warp(image, H, method='bilinear', canvas=None):
    """Return (float pixels, valid mask, (world x origin, world y origin))."""
    image = np.asarray(image, float)
    x0, y0, x1, y1 = bounds(image, H) if canvas is None else canvas
    width, height = x1-x0+1, y1-y0+1
    if width <= 0 or height <= 0 or width*height > 30_000_000:
        raise ValueError('Invalid or excessive canvas; inspect correspondences or resize photos.')
    output = np.zeros((height, width) + image.shape[2:])
    mask = np.zeros((height, width), bool)
    inverse = np.linalg.inv(H)
    ih, iw = image.shape[:2]
    for row in range(0, height, 128):
        yy, xx = np.mgrid[row:min(row+128, height), :width]
        q = np.stack([xx+x0, yy+y0, np.ones_like(xx)], axis=-1) @ inverse.T
        z = q[..., 2]
        safe = np.abs(z) > 1e-12
        sx = np.divide(q[..., 0], z, out=np.full_like(z, -1.), where=safe)
        sy = np.divide(q[..., 1], z, out=np.full_like(z, -1.), where=safe)
        valid = safe & (sx >= -1e-8) & (sy >= -1e-8) & (sx <= iw-1+1e-8) & (sy <= ih-1+1e-8)
        sx, sy = np.clip(sx, 0, iw-1), np.clip(sy, 0, ih-1)
        if method == 'nearest':
            values = image[np.floor(sy+.5).astype(int), np.floor(sx+.5).astype(int)]
        elif method == 'bilinear':
            lx, ly = np.floor(sx).astype(int), np.floor(sy).astype(int)
            rx, ry = np.minimum(lx+1, iw-1), np.minimum(ly+1, ih-1)
            dx, dy = sx-lx, sy-ly
            if image.ndim == 3:
                dx, dy = dx[..., None], dy[..., None]
            values = ((1-dx)*(1-dy)*image[ly, lx] + dx*(1-dy)*image[ly, rx]
                      + (1-dx)*dy*image[ry, lx] + dx*dy*image[ry, rx])
        else:
            raise ValueError('Unknown interpolation method.')
        output[row:row+len(yy)] = values * (valid[..., None] if image.ndim == 3 else valid)
        mask[row:row+len(yy)] = valid
    return output, mask, (x0, y0)


def warpImageNearestNeighbor(im, H):
    return warp(im, H, 'nearest')[0]


def warpImageBilinear(im, H):
    return warp(im, H, 'bilinear')[0]


def blend(images, homographies):
    boxes = np.array([bounds(im, H) for im, H in zip(images, homographies)])
    canvas = (*boxes[:, :2].min(axis=0), *boxes[:, 2:].max(axis=0))
    total = weights = None
    for im, H in zip(images, homographies):
        h, w = im.shape[:2]
        y, x = np.mgrid[:h, :w]
        feather = np.minimum.reduce([x+1, w-x, y+1, h-y]).astype(float)
        pixels, valid, origin = warp(im, H, canvas=canvas)
        alpha, _, _ = warp(feather, H, canvas=canvas)
        alpha *= valid
        if total is None:
            total, weights = np.zeros_like(pixels), np.zeros_like(alpha)
        total += pixels * alpha[..., None]
        weights += alpha
    return total / np.maximum(weights[..., None], 1e-12), weights > 0


def read(path):
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert('RGB'), float) / 255


def save(path, pixels, mask=None):
    data = np.uint8(np.clip(pixels, 0, 1)*255 + .5)
    if mask is not None:
        data = np.dstack([data, np.uint8(mask)*255])
    Image.fromarray(data).save(path)


def select_points(source, destination, output, count):
    import matplotlib.pyplot as plt
    points = []
    for path in [source, destination]:
        plt.figure(figsize=(12, 8))
        plt.imshow(read(path))
        plt.title(f'{path.name}: select {count} points in the SAME order; right-click undoes')
        selected = plt.ginput(count, timeout=0)
        plt.close()
        if len(selected) != count:
            raise ValueError('Point selection cancelled.')
        points.append(selected)
    output.write_text(json.dumps({'source': points[0], 'destination': points[1]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    pick = sub.add_parser('points')
    pick.add_argument('source', type=Path)
    pick.add_argument('destination', type=Path)
    pick.add_argument('output', type=Path)
    pick.add_argument('--count', type=int, default=10)
    run = sub.add_parser('run')
    run.add_argument('config', type=Path)
    args = parser.parse_args()
    if args.command == 'points':
        select_points(args.source, args.destination, args.output, args.count)
        return
    config = json.loads(args.config.read_text())
    root = args.config.resolve().parent
    out = root / config.get('output', 'results')
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    for job in config['jobs']:
        name = job['name']
        source = read(root / job['source'])
        pts = json.loads((root / job['points']).read_text())
        H = computeH(pts['source'], pts['destination'])
        A, b = linear_system(pts['source'], pts['destination'])
        error = np.linalg.norm(transform(pts['source'], H)-pts['destination'], axis=1)
        info = {'H': H.tolist(), 'A': A.tolist(), 'b': b.tolist(),
                'reprojection_rmse_pixels': float(np.sqrt(np.mean(error**2)))}
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        destination = read(root / job['reference']) if job['kind'] == 'mosaic' else None
        for ax, im, points in zip(axes, [source, destination], [pts['source'], pts['destination']]):
            if im is not None:
                ax.imshow(im)
            else:
                ax.invert_yaxis()
                ax.set_aspect('equal')
            p = np.asarray(points)
            ax.scatter(p[:, 0], p[:, 1], color='red')
            for i, (x, y) in enumerate(p):
                ax.annotate(str(i+1), (x, y), color='blue')
        fig.savefig(out / f'{name}_points.png', bbox_inches='tight')
        plt.close(fig)
        for method in ['nearest', 'bilinear']:
            start = perf_counter()
            pixels, mask, origin = warp(source, H, method)
            info[method+'_seconds'] = perf_counter()-start
            info['warp_origin_xy'] = origin
            save(out / f'{name}_{method}.png', pixels, mask)
        if destination is not None:
            pixels, mask = blend([source, destination], [H, np.eye(3)])
            save(out / f'{name}_mosaic.png', pixels, mask)
        (out / f'{name}.json').write_text(json.dumps(info, indent=2, default=lambda x: int(x)))
        print(f'{name}: RMSE {info["reprojection_rmse_pixels"]:.3f} pixels; saved to {out}')


if __name__ == '__main__':
    main()
