<script setup lang="ts">
import { computed } from 'vue'
import type { AttributeDistribution } from '@/types'

const props = withDefaults(
  defineProps<{
    distribution: AttributeDistribution | null | undefined
    valueMarker?: number | null
    width?: number
    height?: number
    markerColor?: string
  }>(),
  { width: 110, height: 32, markerColor: '#ef4444' },
)

interface Bar {
  x: number
  width: number
  height: number
}

const bars = computed<Bar[] | null>(() => {
  const d = props.distribution
  if (!d || d.bin_counts.length === 0) return null
  const maxCount = Math.max(...d.bin_counts)
  if (maxCount === 0) return null

  const innerW = props.width
  const innerH = props.height
  const n = d.bin_counts.length
  const barW = innerW / n

  return d.bin_counts.map((c, i) => ({
    x: i * barW,
    width: Math.max(barW - 0.5, 0.5),
    height: (c / maxCount) * innerH,
  }))
})

const markerX = computed<number | null>(() => {
  const d = props.distribution
  const v = props.valueMarker
  if (!d || v == null || d.bin_edges.length < 2) return null
  const lo = d.bin_edges[0]
  const hi = d.bin_edges[d.bin_edges.length - 1]
  if (hi === lo) return null
  const t = (v - lo) / (hi - lo)
  if (!isFinite(t)) return null
  return Math.max(0, Math.min(1, t)) * props.width
})
</script>

<template>
  <svg
    v-if="bars"
    :width="width"
    :height="height + 1"
    :viewBox="`0 0 ${width} ${height + 1}`"
    class="block"
  >
    <g>
      <rect
        v-for="(b, i) in bars"
        :key="i"
        :x="b.x"
        :y="height - b.height"
        :width="b.width"
        :height="b.height"
        fill="currentColor"
        opacity="0.35"
      />
    </g>
    <line
      :x1="0"
      :y1="height + 0.5"
      :x2="width"
      :y2="height + 0.5"
      stroke="currentColor"
      stroke-width="0.5"
      opacity="0.3"
    />
    <line
      v-if="markerX != null"
      :x1="markerX"
      :y1="0"
      :x2="markerX"
      :y2="height"
      :stroke="markerColor"
      stroke-width="1.5"
    />
  </svg>
  <div v-else class="text-[9px] font-mono text-neutral/30" :style="{ height: `${height}px` }">
    no data
  </div>
</template>
