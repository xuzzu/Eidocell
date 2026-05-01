/**
 * useGateWorker — Composable for client-side gate computation via Web Worker.
 *
 * Provides instant visual feedback for gate operations while the backend
 * computes authoritative results.
 */

import { ref, onBeforeUnmount } from 'vue'

export interface GateWorkerRequest {
  gateType: string
  definition: Record<string, unknown>
  xData: Float32Array
  yData: Float32Array | null
}

export function useGateWorker() {
  const insideMask = ref<Uint8Array | null>(null)
  const computing = ref(false)

  let worker: Worker | null = null

  function getWorker(): Worker {
    if (!worker) {
      worker = new Worker(
        new URL('../workers/gateWorker.ts', import.meta.url),
        { type: 'module' },
      )
      worker.onmessage = (e: MessageEvent<{ insideMask: Uint8Array }>) => {
        insideMask.value = e.data.insideMask
        computing.value = false
      }
      worker.onerror = () => {
        computing.value = false
      }
    }
    return worker
  }

  function compute(request: GateWorkerRequest) {
    computing.value = true
    const w = getWorker()
    // Transfer the arrays (zero-copy)
    const transferables: Transferable[] = [request.xData.buffer as ArrayBuffer]
    if (request.yData) transferables.push(request.yData.buffer as ArrayBuffer)
    w.postMessage(request, transferables)
  }

  function terminate() {
    worker?.terminate()
    worker = null
  }

  onBeforeUnmount(() => terminate())

  return {
    insideMask,
    computing,
    compute,
    terminate,
  }
}
