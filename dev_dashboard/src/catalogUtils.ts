import type { CatalogImage } from "./types";

/** Stable React key — metadata.json can repeat the same id for different files. */
export function catalogImageKey(img: CatalogImage, index?: number): string {
  const id = String(img.id ?? "");
  const file = img.filename ?? "";
  if (id && file) return `${id}:${file}`;
  if (id) return id;
  if (file) return file;
  return `catalog-${index ?? 0}`;
}

/** Drop duplicate catalog rows (same id or same id+filename). Keeps first occurrence. */
export function dedupeCatalogImages(images: CatalogImage[]): CatalogImage[] {
  const seen = new Set<string>();
  const out: CatalogImage[] = [];
  for (let i = 0; i < images.length; i++) {
    const img = images[i];
    const key = catalogImageKey(img, i);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(img);
  }
  return out;
}
