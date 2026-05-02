import { onBeforeUnmount, ref, shallowRef } from 'vue'
import { wsUrl } from '@/api/client'

export interface PreviewTile {
  sampleId: string
  ok: boolean
  url: string | null         // Blob URL for the rendered overlay PNG
  attrs: Record<string, number> | null
}

export interface PreviewRequest {
  method: string
  params: Record<string, number>
  sampleIds: string[]
}

interface FrameHeader {
  request_id: number
  sample_id: string
  ok: boolean
  attrs: Record<string, number> | null
}

interface DoneMsg {
  done: true
  request_id: number
}

interface ErrorMsg {
  error: string
  request_id?: number
}

/**
 * Streaming segmentation preview client.
 *
 * - Maintains one persistent WebSocket per session.
 * - Each call to `request(...)` cancels any in-flight stream server-side and
 *   starts a new one.
 * - Tiles are exposed as a reactive Map<sampleId, PreviewTile>; each tile
 *   updates as its overlay frame arrives, so the UI fills in progressively.
 * - Stale frames (from superseded requests) are dropped.
 */
export function useSegmentationPreviewWs(getSessionId: () => string | null) {
  const tiles = shallowRef(new Map<string, PreviewTile>())
  const inFlight = ref(false)
  const error = ref<string | null>(null)

  let socket: WebSocket | null = null
  let connectPromise: Promise<WebSocket> | null = null
  let nextRequestId = 1
  let activeRequestId = 0
  // Blob URLs we own and need to revoke when superseded
  const ownedUrls = new Set<string>()

  function revokeAll() {
    for (const u of ownedUrls) URL.revokeObjectURL(u)
    ownedUrls.clear()
  }

  function setTile(sampleId: string, tile: PreviewTile) {
    const next = new Map(tiles.value)
    const existing = next.get(sampleId)
    if (existing?.url && ownedUrls.has(existing.url)) {
      URL.revokeObjectURL(existing.url)
      ownedUrls.delete(existing.url)
    }
    if (tile.url) ownedUrls.add(tile.url)
    next.set(sampleId, tile)
    tiles.value = next
  }

  function clear() {
    revokeAll()
    tiles.value = new Map()
  }

  async function ensureConnected(): Promise<WebSocket> {
    if (socket && socket.readyState === WebSocket.OPEN) return socket
    if (connectPromise) return connectPromise

    const sid = getSessionId()
    if (!sid) throw new Error('No active session')
    const url = wsUrl(`/sessions/${sid}/segmentation/preview/ws`)

    connectPromise = new Promise((resolve, reject) => {
      const ws = new WebSocket(url)
      ws.binaryType = 'arraybuffer'

      ws.onopen = () => {
        socket = ws
        connectPromise = null
        resolve(ws)
      }

      ws.onerror = () => {
        connectPromise = null
        reject(new Error('WebSocket connection failed'))
      }

      ws.onclose = () => {
        if (socket === ws) socket = null
        inFlight.value = false
      }

      ws.onmessage = (ev) => {
        if (typeof ev.data === 'string') {
          try {
            const msg = JSON.parse(ev.data) as DoneMsg | ErrorMsg
            if ('done' in msg && msg.request_id === activeRequestId) {
              inFlight.value = false
            } else if ('error' in msg) {
              error.value = msg.error
              if (msg.request_id === activeRequestId) inFlight.value = false
            }
          } catch {
            // ignore
          }
          return
        }

        // Binary frame: [4-byte BE header length][header JSON][PNG bytes]
        const buf = ev.data as ArrayBuffer
        const view = new DataView(buf)
        if (buf.byteLength < 4) return
        const headerLen = view.getUint32(0, false)
        if (buf.byteLength < 4 + headerLen) return
        const headerBytes = new Uint8Array(buf, 4, headerLen)
        const header = JSON.parse(new TextDecoder().decode(headerBytes)) as FrameHeader
        if (header.request_id !== activeRequestId) return // stale

        const pngBytes = new Uint8Array(buf, 4 + headerLen)
        let url: string | null = null
        if (header.ok && pngBytes.byteLength > 0) {
          url = URL.createObjectURL(new Blob([pngBytes], { type: 'image/png' }))
        }
        setTile(header.sample_id, {
          sampleId: header.sample_id,
          ok: header.ok,
          url,
          attrs: header.attrs,
        })
      }
    })
    return connectPromise
  }

  async function request(req: PreviewRequest) {
    if (req.sampleIds.length === 0) return
    error.value = null
    const ws = await ensureConnected()
    activeRequestId = nextRequestId++
    inFlight.value = true
    ws.send(JSON.stringify({
      request_id: activeRequestId,
      method: req.method,
      params: req.params,
      sample_ids: req.sampleIds,
    }))
  }

  function close() {
    if (socket) {
      socket.close()
      socket = null
    }
    revokeAll()
  }

  onBeforeUnmount(() => {
    close()
  })

  return {
    tiles,
    inFlight,
    error,
    request,
    clear,
    close,
  }
}
