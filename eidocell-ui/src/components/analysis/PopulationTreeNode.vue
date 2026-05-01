<script setup lang="ts">
import { computed } from 'vue'
import { Eye, EyeOff, ChevronRight, ChevronDown, GitBranchPlus } from 'lucide-vue-next'
import type { GateTreeNode } from '@/api/analysis'

const props = defineProps<{
  node: GateTreeNode
  depth: number
  expandedIds: Set<string>
  parentGateId: string | null
}>()

const emit = defineEmits<{
  toggleExpand: [id: string]
  toggleActive: [id: string, currentActive: boolean]
  select: [id: string]
  setParent: [id: string]
}>()

const hasChildren = computed(() => props.node.children.length > 0)
const isExpanded = computed(() => props.expandedIds.has(props.node.id))
const isActiveParent = computed(() => props.parentGateId === props.node.id)
const indent = computed(() => ({ paddingLeft: `${props.depth * 16 + 4}px` }))
</script>

<template>
  <div>
    <div
      class="flex items-center gap-1.5 py-1 px-1 rounded-[2px] cursor-pointer group transition-colors"
      :class="isActiveParent ? 'bg-info/10 border border-info/20' : 'hover:bg-base-200'"
      :style="indent"
      @click="emit('select', node.id)"
    >
      <!-- Expand/collapse chevron -->
      <button
        v-if="hasChildren"
        class="w-4 h-4 flex items-center justify-center shrink-0 text-neutral/30"
        @click.stop="emit('toggleExpand', node.id)"
      >
        <ChevronDown v-if="isExpanded" class="w-3 h-3 stroke-[2px]" />
        <ChevronRight v-else class="w-3 h-3 stroke-[2px]" />
      </button>
      <div v-else class="w-4 shrink-0"></div>

      <!-- Color dot -->
      <span class="w-2 h-2 rounded-full shrink-0" :style="{ backgroundColor: node.color }"></span>

      <!-- Name + stats -->
      <div class="flex-1 min-w-0 flex items-center gap-1.5">
        <span class="text-[10px] font-mono font-bold truncate">{{ node.name }}</span>
        <span class="text-[9px] font-mono text-neutral/40 shrink-0">
          {{ node.sample_count.toLocaleString() }} · {{ node.percentage.toFixed(1) }}%
        </span>
      </div>

      <!-- Sub-gate within this gate -->
      <button
        class="h-5 w-5 flex items-center justify-center rounded-[2px] transition-colors shrink-0 opacity-0 group-hover:opacity-100"
        :class="isActiveParent ? 'text-info opacity-100' : 'text-neutral/40 hover:text-info'"
        :title="isActiveParent ? 'Clear sub-gating' : 'Sub-gate within this population'"
        @click.stop="emit('setParent', node.id)"
      >
        <GitBranchPlus class="w-3 h-3 stroke-[2px]" />
      </button>

      <!-- Active toggle -->
      <button
        class="h-5 w-5 flex items-center justify-center rounded-[2px] transition-colors shrink-0"
        :class="node.is_active ? 'text-neutral' : 'text-neutral/25'"
        @click.stop="emit('toggleActive', node.id, node.is_active)"
      >
        <Eye v-if="node.is_active" class="w-3 h-3 stroke-[2px]" />
        <EyeOff v-else class="w-3 h-3 stroke-[2px]" />
      </button>
    </div>

    <!-- Children (recursive) -->
    <template v-if="hasChildren && isExpanded">
      <PopulationTreeNode
        v-for="child in node.children"
        :key="child.id"
        :node="child"
        :depth="depth + 1"
        :expanded-ids="expandedIds"
        :parent-gate-id="parentGateId"
        @toggle-expand="(id) => emit('toggleExpand', id)"
        @toggle-active="(id, active) => emit('toggleActive', id, active)"
        @select="(id) => emit('select', id)"
        @set-parent="(id) => emit('setParent', id)"
      />
    </template>
  </div>
</template>

<script lang="ts">
export default { name: 'PopulationTreeNode' }
</script>
