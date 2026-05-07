<script setup lang="ts">
// @ts-ignore — vue-grid-layout-v3 has no type declarations
import { GridLayout, GridItem } from 'vue-grid-layout-v3'
import { useAnalysisStore } from '@/stores/analysis'
import PlotWidget from '@/components/analysis/PlotWidget.vue'
import type { GateCreate, PlotLayout } from '@/types'

const analysis = useAnalysisStore()

const emit = defineEmits<{
  gateCreated: [plotId: string, gate: GateCreate]
  gateUpdated: [gateId: string, definition: Record<string, unknown>]
  gateDeleted: [gateId: string]
}>()

function getPlot(plotId: string) {
  return analysis.plots.find(p => p.id === plotId)
}

function onLayoutUpdated(newLayout: PlotLayout[]) {
  analysis.plotLayouts = newLayout
}

function onRemovePlot(plotId: string) {
  analysis.removePlotFromWorkspace(plotId)
}
</script>

<template>
  <GridLayout
    :layout="analysis.plotLayouts"
    :col-num="12"
    :row-height="60"
    :is-draggable="true"
    :is-resizable="true"
    :margin="[8, 8]"
    :use-css-transforms="true"
    :vertical-compact="true"
    @layout-updated="onLayoutUpdated"
  >
    <GridItem
      v-for="item in analysis.plotLayouts"
      :key="item.i"
      :i="item.i"
      :x="item.x"
      :y="item.y"
      :w="item.w"
      :h="item.h"
      :min-w="3"
      :min-h="3"
      drag-allow-from=".widget-drag-handle"
      drag-ignore-from="button, input, svg, .gate-tool"
    >
      <PlotWidget
        v-if="getPlot(item.i)"
        :plot="getPlot(item.i)!"
        :plot-data="analysis.plotDataCache.get(item.i) ?? null"
        :gates="analysis.gatesByPlotId.get(item.i) ?? []"
        :is-active="analysis.selectedPlotId === item.i"
        @select="analysis.selectPlot(item.i)"
        @gate-created="(g) => emit('gateCreated', item.i, g)"
        @gate-updated="(id, def) => emit('gateUpdated', id, def)"
        @gate-deleted="(id) => emit('gateDeleted', id)"
        @remove="onRemovePlot(item.i)"
      />
    </GridItem>
  </GridLayout>
</template>

<style>
/* vue-grid-layout-v3 styles (inlined because the package's CSS export is broken) */
.vue-grid-layout {
  position: relative;
  transition: height 0.2s ease;
}
.vue-grid-item {
  transition: all 0.2s ease;
  transition-property: left, top, right;
}
.vue-grid-item.no-touch {
  touch-action: none;
}
.vue-grid-item.cssTransforms {
  transition-property: transform;
  left: 0;
  right: auto;
}
.vue-grid-item.resizing {
  opacity: 0.6;
  z-index: 3;
}
.vue-grid-item.vue-draggable-dragging {
  transition: none;
  z-index: 3;
}
.vue-grid-item.vue-grid-placeholder {
  background: oklch(var(--n));
  opacity: 0.08;
  border-radius: 2px;
  transition-duration: 0.1s;
  z-index: 2;
  user-select: none;
}
.vue-grid-item > .vue-resizable-handle {
  position: absolute;
  width: 20px;
  height: 20px;
  bottom: 0;
  right: 0;
  padding: 0 3px 3px 0;
  background-repeat: no-repeat;
  background-origin: content-box;
  box-sizing: border-box;
  cursor: se-resize;
  opacity: 0;
  transition: opacity 0.15s ease;
}
.vue-grid-item:hover > .vue-resizable-handle {
  opacity: 0.3;
}
.vue-grid-item.disable-userselect {
  user-select: none;
}

/* Ensure grid items fill their container */
.vue-grid-item > div {
  height: 100%;
}
</style>
