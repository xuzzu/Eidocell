/**
 * Pinia plugin that mirrors selected stores across Electron BrowserWindows
 * via IPC. Every state change in an opted-in store is debounced, serialized,
 * and rebroadcast to other windows; incoming snapshots are applied via
 * `$patch` while a per-store flag suppresses re-emission to avoid loops.
 *
 * Sets/Maps inside state are encoded so JSON serialization preserves them.
 * Excluded keys (per-store) skip the wire — used for non-serializable refs
 * (function hooks) and window-local UI caches (large data, animation flashes).
 */
import type { PiniaPluginContext, StateTree } from 'pinia'
import { toRaw } from 'vue'

const DEBOUNCE_MS = 10

interface SyncConfig {
  exclude?: string[]
}

const SYNC_STORES: Record<string, SyncConfig> = {
  session: {},
  popouts: {},
  gallery: {
    exclude: ['openSimilarityDialog', 'loading'],
  },
  classes: {
    exclude: ['loading'],
  },
  clusters: {
    exclude: ['recentlyAddedIds', 'recentlyRemovedIds', 'loading'],
  },
  segmentation: {
    exclude: ['loading'],
  },
  analysis: {
    // Plot data cache is window-local — each window re-fetches as needed.
    exclude: ['plotDataCache', 'loading'],
  },
}

const SET_TAG = '__set__'
const MAP_TAG = '__map__'

function encode(value: unknown): unknown {
  if (value === null || value === undefined) return value
  if (value instanceof Set) {
    return { [SET_TAG]: Array.from(value).map(encode) }
  }
  if (value instanceof Map) {
    return { [MAP_TAG]: Array.from(value.entries()).map(([k, v]) => [encode(k), encode(v)]) }
  }
  if (Array.isArray(value)) return value.map(encode)
  if (typeof value === 'function') return undefined
  if (typeof value === 'object') {
    const out: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      const enc = encode(v)
      if (enc !== undefined) out[k] = enc
    }
    return out
  }
  return value
}

function decode(value: unknown): unknown {
  if (value === null || value === undefined) return value
  if (typeof value !== 'object') return value
  if (Array.isArray(value)) return value.map(decode)
  const obj = value as Record<string, unknown>
  if (SET_TAG in obj && Array.isArray(obj[SET_TAG])) {
    return new Set((obj[SET_TAG] as unknown[]).map(decode))
  }
  if (MAP_TAG in obj && Array.isArray(obj[MAP_TAG])) {
    return new Map((obj[MAP_TAG] as [unknown, unknown][]).map(([k, v]) => [decode(k), decode(v)]))
  }
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(obj)) {
    out[k] = decode(v)
  }
  return out
}

function pickSerializable(state: StateTree, exclude: string[] = []): StateTree {
  const excludeSet = new Set(exclude)
  const out: StateTree = {}
  for (const [k, v] of Object.entries(state)) {
    if (excludeSet.has(k)) continue
    if (typeof v === 'function') continue
    const enc = encode(toRaw(v))
    if (enc !== undefined) out[k] = enc
  }
  return out
}

function applyDecodedPatch(state: StateTree, patch: StateTree, exclude: string[] = []) {
  const excludeSet = new Set(exclude)
  for (const [k, v] of Object.entries(patch)) {
    if (excludeSet.has(k)) continue
    state[k] = decode(v) as never
  }
}

export function windowSyncPlugin(ctx: PiniaPluginContext) {
  const ipc = (typeof window !== 'undefined' ? (window as any).ipcRenderer : null) as
    | {
        invoke: (channel: string, ...args: unknown[]) => Promise<unknown>
        send: (channel: string, ...args: unknown[]) => void
        on: (channel: string, listener: (event: unknown, ...args: unknown[]) => void) => void
      }
    | null
  if (!ipc) return

  const config = SYNC_STORES[ctx.store.$id]
  if (!config) return

  const storeId = ctx.store.$id
  const exclude = config.exclude ?? []
  let applyingRemote = false
  let pendingTimer: ReturnType<typeof setTimeout> | null = null
  // Once a live sync:apply has been received, drop any in-flight initial
  // snapshot — the live frame is fresher than the cached one.
  let liveApplyReceived = false

  // Hydrate from main-process snapshot cache on boot.
  ipc.invoke('sync:snapshot', storeId).then((snapshot) => {
    if (liveApplyReceived) return
    if (snapshot && typeof snapshot === 'object') {
      applyingRemote = true
      ctx.store.$patch((state) => applyDecodedPatch(state, snapshot as StateTree, exclude))
      queueMicrotask(() => { applyingRemote = false })
    }
  })

  // Receive remote updates.
  ipc.on('sync:apply', (_event, ...args: unknown[]) => {
    const [incomingStoreId, payload] = args as [string, StateTree]
    if (incomingStoreId !== storeId) return
    liveApplyReceived = true
    applyingRemote = true
    ctx.store.$patch((state) => applyDecodedPatch(state, payload, exclude))
    queueMicrotask(() => { applyingRemote = false })
  })

  // Broadcast local mutations.
  ctx.store.$subscribe(
    () => {
      if (applyingRemote) return
      if (pendingTimer) clearTimeout(pendingTimer)
      pendingTimer = setTimeout(() => {
        pendingTimer = null
        const serialized = pickSerializable(toRaw(ctx.store.$state), exclude)
        ipc.send('sync:broadcast', storeId, serialized)
      }, DEBOUNCE_MS)
    },
    { detached: true },
  )
}
