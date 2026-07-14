// Convert between frontend TestCase shape (user_images / ducon_images) and the
// backend persistence shape (inputs[] with source upload|catalog).
import type { CatalogImage, TestCase, TestCaseImage } from "./types";

type RawTestCase = Record<string, unknown>;

function uploadUrl(uploadId: string, name?: string): string {
  const ext = name?.includes(".") ? name.split(".").pop() : "png";
  return `/dev/uploads/${uploadId}.${ext ?? "png"}`;
}

function catalogUrlFromMeta(meta: Record<string, unknown>): string {
  const direct = String(meta.full_url ?? meta.thumb_url ?? meta.url ?? "");
  if (direct) return direct;
  const filename = String(meta.filename ?? "");
  if (filename) return `/public/images/${filename}`;
  return "";
}

function resolveCatalogUrl(
  meta: Record<string, unknown>,
  catalogImages?: CatalogImage[],
): string {
  const fromMeta = catalogUrlFromMeta(meta);
  if (fromMeta) return fromMeta;
  const catalogId = meta.catalog_id ?? meta.id;
  if (catalogId == null || !catalogImages?.length) return "";
  const match = catalogImages.find((c) => String(c.id) === String(catalogId));
  if (!match) return "";
  return match.full_url || match.thumb_url || catalogUrlFromMeta({ filename: match.filename });
}

/** Normalize any stored/API test case into the frontend TestCase shape. */
export function normalizeTestCase(
  raw: RawTestCase,
  catalogImages?: CatalogImage[],
): TestCase {
  const id = String(raw.id ?? "");
  const name = String(raw.name ?? "");
  const use_ducon_data = Boolean(raw.use_ducon_data);
  const hint = raw.hint ? String(raw.hint) : undefined;
  const created_at = raw.created_at ? String(raw.created_at) : undefined;

  // Already frontend format — ensure arrays exist and enrich catalog URLs.
  if (Array.isArray(raw.user_images) || Array.isArray(raw.ducon_images)) {
    const enrich = (images: TestCaseImage[]) =>
      images.map((im) => {
        if (im.url) return im;
        const meta = im.metadata ?? {};
        const url = resolveCatalogUrl(meta, catalogImages);
        return url ? { ...im, url } : im;
      });
    return {
      id,
      name,
      user_images: enrich((raw.user_images as TestCaseImage[]) ?? []),
      ducon_images: enrich((raw.ducon_images as TestCaseImage[]) ?? []),
      use_ducon_data,
      hint,
      created_at,
    };
  }

  // Backend format: inputs[]
  const user_images: TestCaseImage[] = [];
  const ducon_images: TestCaseImage[] = [];

  for (const inp of (raw.inputs as RawTestCase[]) ?? []) {
    const role = String(inp.role ?? "user");
    const source = String(inp.source ?? "upload");
    const label = String(inp.label ?? inp.name ?? "image");
    const meta = (inp.metadata as Record<string, unknown>) ?? {};

    let url = "";
    if (source === "upload" && inp.upload_id) {
      url = uploadUrl(String(inp.upload_id), String(inp.name ?? meta.filename ?? ""));
    } else if (source === "catalog") {
      url = resolveCatalogUrl(
        {
          ...meta,
          catalog_id: inp.catalog_id ?? meta.catalog_id ?? meta.id,
          filename: meta.filename ?? inp.name,
        },
        catalogImages,
      );
    }

    const img: TestCaseImage = {
      url,
      role: role === "user" ? "user" : "ducon",
      metadata: {
        ...meta,
        label,
        source,
        upload_id: inp.upload_id,
        catalog_id: inp.catalog_id ?? meta.catalog_id ?? meta.id,
        name: inp.name ?? meta.name,
        filename: meta.filename ?? inp.name,
        full_url: meta.full_url ?? url,
        thumb_url: meta.thumb_url ?? url,
      },
    };

    if (role === "user" || source === "upload") {
      user_images.push(img);
    } else {
      ducon_images.push(img);
    }
  }

  return { id, name, user_images, ducon_images, use_ducon_data, hint, created_at };
}

/** Serialize a frontend TestCase for the backend /dev/test-cases API. */
export function toBackendTestCase(tc: TestCase): RawTestCase {
  const inputs: RawTestCase[] = [];

  for (const im of tc.user_images ?? []) {
    const meta = im.metadata ?? {};
    const uploadId = meta.upload_id as string | undefined;
    inputs.push({
      label: (meta.label as string) ?? (meta.filename as string) ?? "user image",
      role: "user",
      source: "upload",
      upload_id: uploadId,
      name: (meta.filename as string) ?? (meta.name as string),
      metadata: {
        ...meta,
        filename: meta.filename ?? meta.name,
        full_url: im.url || meta.full_url,
      },
    });
  }

  for (const im of tc.ducon_images ?? []) {
    const meta = im.metadata ?? {};
    const catalogId = meta.catalog_id ?? meta.id;
    const filename = meta.filename as string | undefined;
    inputs.push({
      label: (meta.name as string) ?? (meta.label as string) ?? "ducon image",
      role: (meta.type as string) ?? "design",
      source: "catalog",
      catalog_id: catalogId,
      name: meta.name ?? filename,
      metadata: {
        ...meta,
        catalog_id: catalogId,
        filename,
        full_url: im.url || meta.full_url,
        thumb_url: meta.thumb_url ?? im.url,
      },
    });
  }

  return {
    id: tc.id,
    name: tc.name,
    hint: tc.hint,
    use_ducon_data: tc.use_ducon_data,
    inputs,
  };
}
