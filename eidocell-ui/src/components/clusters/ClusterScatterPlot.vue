<script setup lang="ts">
import { computed, ref, onMounted, onUnmounted, watch } from 'vue'
import { useClustersStore } from '@/stores/clusters'

const clusters = useClustersStore()
const canvasRef = ref<HTMLCanvasElement | null>(null)
const containerRef = ref<HTMLDivElement | null>(null)

const hasData = computed(() => clusters.embeddings.length > 0)

function draw() {
  const canvas = canvasRef.value
  if (!canvas || !hasData.value) return

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const container = containerRef.value
  if (!container) return

  const dpr = window.devicePixelRatio || 1
  const rect = container.getBoundingClientRect()
  canvas.width = rect.width * dpr
  canvas.height = rect.height * dpr
  canvas.style.width = `${rect.width}px`
  canvas.style.height = `${rect.height}px`
  ctx.scale(dpr, dpr)

  const w = rect.width
  const h = rect.height
  const padding = 32

  // Clear
  ctx.clearRect(0, 0, w, h)

  // Compute bounds
  const pts = clusters.embeddings
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
  for (const p of pts) {
    if (p.x < minX) minX = p.x
    if (p.x > maxX) maxX = p.x
    if (p.y < minY) minY = p.y
    if (p.y > maxY) maxY = p.y
  }

  const rangeX = maxX - minX || 1
  const rangeY = maxY - minY || 1

  function toScreenX(x: number) {
    return padding + ((x - minX) / rangeX) * (w - 2 * padding)
  }
  function toScreenY(y: number) {
    return padding + ((y - minY) / rangeY) * (h - 2 * padding)
  }

  // Draw background grid
  ctx.strokeStyle = 'rgba(128, 128, 128, 0.1)'
  ctx.lineWidth = 1
  for (let i = 0; i <= 4; i++) {
    const gx = padding + (i / 4) * (w - 2 * padding)
    const gy = padding + (i / 4) * (h - 2 * padding)
    ctx.beginPath()
    ctx.moveTo(gx, padding)
    ctx.lineTo(gx, h - padding)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(padding, gy)
    ctx.lineTo(w - padding, gy)
    ctx.stroke()
  }

  // Draw points
  const radius = Math.max(2, Math.min(5, 400 / Math.sqrt(pts.length)))
  for (const p of pts) {
    const sx = toScreenX(p.x)
    const sy = toScreenY(p.y)
    ctx.beginPath()
    ctx.arc(sx, sy, radius, 0, Math.PI * 2)
    ctx.fillStyle = p.cluster_color || '#888888'
    ctx.globalAlpha = 0.75
    ctx.fill()
    ctx.globalAlpha = 1
  }
}

let resizeObserver: ResizeObserver | null = null

onMounted(() => {
  if (containerRef.value) {
    resizeObserver = new ResizeObserver(() => draw())
    resizeObserver.observe(containerRef.value)
  }
  draw()
})

onUnmounted(() => {
  resizeObserver?.disconnect()
})

watch(() => clusters.embeddings, () => draw(), { deep: true })
</script>

<template>
  <div
    ref="containerRef"
    class="w-full h-full min-h-[200px] bg-base-200/50 rounded-xl relative"
  >
    <canvas
      v-if="hasData"
      ref="canvasRef"
      class="block w-full h-full"
    />
    <div
      v-else
      class="absolute inset-0 flex items-center justify-center text-base-content/40 text-sm"
    >
      Run clustering to see 2D projection
    </div>
  </div>
</template>
