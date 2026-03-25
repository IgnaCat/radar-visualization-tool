const tileNativeZoomCache = new Map();

export function setTileNativeZoomMetadata(tilejsonUrl, metadata) {
  if (!tilejsonUrl || !metadata) return;
  tileNativeZoomCache.set(tilejsonUrl, metadata);
}

export function getTileNativeZoomMetadata(tilejsonUrl) {
  if (!tilejsonUrl) return null;
  return tileNativeZoomCache.get(tilejsonUrl) || null;
}
