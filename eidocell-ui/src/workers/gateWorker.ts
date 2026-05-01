/**
 * Web Worker for client-side point-in-gate computation.
 *
 * Runs gate tests on Float32Arrays off the main thread for instant visual
 * feedback while the backend computes authoritative results.
 *
 * Message format:
 *   { gateType, definition, xData: Float32Array, yData: Float32Array | null }
 * Response:
 *   { insideMask: Uint8Array }  (1 = inside gate, 0 = outside)
 */

interface GateMessage {
  gateType: string
  definition: Record<string, unknown>
  xData: Float32Array
  yData: Float32Array | null
}

self.onmessage = (e: MessageEvent<GateMessage>) => {
  const { gateType, definition, xData, yData } = e.data
  const n = xData.length
  const result = new Uint8Array(n)

  if (gateType === 'rectangular') {
    const { x, y, width, height } = definition as {
      x: number; y: number; width: number; height: number
    }
    const x1 = x + width
    const y1 = y + height
    for (let i = 0; i < n; i++) {
      const px = xData[i]
      const py = yData![i]
      result[i] = (px >= x && px <= x1 && py >= y && py <= y1) ? 1 : 0
    }
  } else if (gateType === 'interval') {
    const { min, max } = definition as { min: number; max: number }
    for (let i = 0; i < n; i++) {
      result[i] = (xData[i] >= min && xData[i] <= max) ? 1 : 0
    }
  } else if (gateType === 'polygon') {
    const vertices = definition.vertices as number[][]
    for (let i = 0; i < n; i++) {
      result[i] = pointInPolygon(xData[i], yData![i], vertices) ? 1 : 0
    }
  } else if (gateType === 'ellipse') {
    const { cx, cy, rx, ry } = definition as {
      cx: number; cy: number; rx: number; ry: number
    }
    const angle = ((definition.angle as number) ?? 0) * Math.PI / 180
    const cosA = Math.cos(angle)
    const sinA = Math.sin(angle)
    const rxSq = rx * rx
    const rySq = ry * ry
    for (let i = 0; i < n; i++) {
      const dx = xData[i] - cx
      const dy = yData![i] - cy
      const nx = cosA * dx + sinA * dy
      const ny = -sinA * dx + cosA * dy
      result[i] = ((nx * nx) / rxSq + (ny * ny) / rySq <= 1) ? 1 : 0
    }
  } else if (gateType === 'quadrant') {
    const { x_threshold, y_threshold } = definition as {
      x_threshold: number; y_threshold: number
    }
    for (let i = 0; i < n; i++) {
      result[i] = (xData[i] >= x_threshold && yData![i] >= y_threshold) ? 1 : 0
    }
  }

  self.postMessage({ insideMask: result }, [result.buffer] as any)
}

/** Ray-casting point-in-polygon algorithm. */
function pointInPolygon(x: number, y: number, vertices: number[][]): boolean {
  const n = vertices.length
  let inside = false
  let j = n - 1
  for (let i = 0; i < n; i++) {
    const xi = vertices[i][0], yi = vertices[i][1]
    const xj = vertices[j][0], yj = vertices[j][1]
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
      inside = !inside
    }
    j = i
  }
  return inside
}
